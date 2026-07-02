"""
PPO scaffolding for the Wordle project.

The full PPO scaffolding copied from the notebook (ReplayMemory, PPOAgent, PPOTrainer,
GAE, the objective terms, scheduler, args). The Wordle environment itself lives in
wordle_env.py; importing it here registers the "Wordle-v0" gym id.

Left as a stub for you to implement (analogous to the per-mode networks you wrote in the
notebook): get_actor_and_critic_wordle -> belief-state MLP + (embedding) action head.
"""

import itertools
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import wandb
from jaxtyping import Bool, Float, Int
from numpy.random import Generator
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from part1_intro_to_rl.utils import set_global_seeds
from part21_dqn.solutions import get_episode_data_from_infos
from part3_ppo import wordle_env  # noqa: F401 — imported for its side effect: registers "Wordle-v0"
from rl_utils import make_env

warnings.filterwarnings("ignore")

chapter = "chapter2_rl"
section = "part3_ppo"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
section_dir = root_dir / chapter / "exercises" / section

Arr = np.ndarray
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


# ======================================================================================
#  PPO SCAFFOLDING (copied from the notebook)
# ======================================================================================


@dataclass
class PPOArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "Wordle-v0"

    # Wandb / logging
    use_wandb: bool = False
    video_log_freq: int | None = None
    wandb_project_name: str = "PPOWordle"
    wandb_entity: str = None

    # Duration of different phases
    total_timesteps: int = 500_000
    num_envs: int = 4
    num_steps_per_rollout: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 4

    # Optimization hyperparameters
    lr: float = 2.5e-4
    max_grad_norm: float = 0.5

    # RL hyperparameters
    gamma: float = 0.99

    # PPO-specific hyperparameters
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.25

    def __post_init__(self):
        self.batch_size = self.num_steps_per_rollout * self.num_envs

        assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.total_phases = self.total_timesteps // self.batch_size
        self.total_training_steps = self.total_phases * self.batches_per_learning_phase * self.num_minibatches

        self.video_save_path = section_dir / "videos"


def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_actor_and_critic(envs: gym.vector.SyncVectorEnv) -> tuple[nn.Module, nn.Module]:
    """Returns (actor, critic), the networks used for PPO."""
    num_obs = np.array(envs.single_observation_space.shape).prod()
    num_actions = envs.single_action_space.n
    actor, critic = get_actor_and_critic_wordle(num_obs, num_actions)
    return actor.to(device), critic.to(device)


def get_actor_and_critic_wordle(num_obs: int, num_actions: int) -> tuple[nn.Module, nn.Module]:
    """
    TODO: your belief-state actor/critic. Starting point we discussed:
        - shared/torso MLP over the belief-state observation (2 layers, ~256, Tanh)
        - critic head -> scalar
        - actor head over the ~13k vocab. Either a plain Linear(hidden, num_actions),
          or the embedding/dot-product head (state_query . word_vec) for cross-word
          credit sharing (needs the vocabulary — grab it from a module constant or a
          closure over the env's allowed_guesses).
    """
    raise NotImplementedError


@t.inference_mode()
def compute_advantages(
    next_value: Float[Tensor, "num_envs"],
    next_terminated: Bool[Tensor, "num_envs"],
    rewards: Float[Tensor, "buffer_size num_envs"],
    values: Float[Tensor, "buffer_size num_envs"],
    terminated: Bool[Tensor, "buffer_size num_envs"],
    gamma: float,
    gae_lambda: float,
) -> Float[Tensor, "buffer_size num_envs"]:
    """Compute advantages using Generalized Advantage Estimation."""
    next_terminated = t.cat((terminated, next_terminated.unsqueeze(0)), dim=0)[1:, :].float()
    next_values = t.cat((values, next_value.unsqueeze(0)), dim=0)[1:, :]
    residuals = rewards + (1 - next_terminated) * gamma * next_values - values
    advantages = residuals
    for s in reversed(range(advantages.size(0) - 1)):
        advantages[s, :] += (1 - next_terminated[s, :]) * gamma * gae_lambda * advantages[s + 1, :]
    return advantages


def get_minibatch_indices(rng: Generator, batch_size: int, minibatch_size: int) -> list[np.ndarray]:
    """Return a list of length `num_minibatches`, each an array of `minibatch_size` indices,
    whose union is [0, ..., batch_size - 1]."""
    assert batch_size % minibatch_size == 0
    num_minibatches = batch_size // minibatch_size
    indices = np.split(rng.permutation(batch_size), num_minibatches)
    return indices


@dataclass
class ReplayMinibatch:
    """Samples from the replay memory. Data ~ (s_t, a_t, logpi(a_t|s_t), A_t, A_t + V(s_t), d_{t+1})."""

    obs: Float[Tensor, " minibatch_size *obs_shape"]
    actions: Int[Tensor, " minibatch_size *action_shape"]
    logprobs: Float[Tensor, " minibatch_size"]
    advantages: Float[Tensor, " minibatch_size"]
    returns: Float[Tensor, " minibatch_size"]
    terminated: Bool[Tensor, " minibatch_size"]


class ReplayMemory:
    """Contains buffer; samples from it to return ReplayMinibatch objects."""

    rng: Generator
    obs: Float[Arr, " buffer_size num_envs *obs_shape"]
    actions: Int[Arr, " buffer_size num_envs *action_shape"]
    logprobs: Float[Arr, " buffer_size num_envs"]
    values: Float[Arr, " buffer_size num_envs"]
    rewards: Float[Arr, " buffer_size num_envs"]
    terminated: Bool[Arr, " buffer_size num_envs"]

    def __init__(
        self,
        num_envs: int,
        obs_shape: tuple,
        action_shape: tuple,
        batch_size: int,
        minibatch_size: int,
        batches_per_learning_phase: int,
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.batches_per_learning_phase = batches_per_learning_phase
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Resets all stored experiences, ready for new ones to be added to memory."""
        self.obs = np.empty((0, self.num_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty((0, self.num_envs, *self.action_shape), dtype=np.int32)
        self.logprobs = np.empty((0, self.num_envs), dtype=np.float32)
        self.values = np.empty((0, self.num_envs), dtype=np.float32)
        self.rewards = np.empty((0, self.num_envs), dtype=np.float32)
        self.terminated = np.empty((0, self.num_envs), dtype=bool)

    def add(
        self,
        obs: Float[Arr, " num_envs *obs_shape"],
        actions: Int[Arr, " num_envs *action_shape"],
        logprobs: Float[Arr, " num_envs"],
        values: Float[Arr, " num_envs"],
        rewards: Float[Arr, " num_envs"],
        terminated: Bool[Arr, " num_envs"],
    ) -> None:
        """Add a batch of transitions to the replay memory."""
        for data, expected_shape in zip(
            [obs, actions, logprobs, values, rewards, terminated],
            [self.obs_shape, self.action_shape, (), (), (), ()],
        ):
            assert isinstance(data, np.ndarray), "data in replay memory must be ndarray"
            assert data.shape == (
                self.num_envs,
                *expected_shape,
            ), f"data added to replay memory was not expected shape {expected_shape}"

        self.obs = np.concatenate((self.obs, obs[None, :]))
        self.actions = np.concatenate((self.actions, actions[None, :]))
        self.logprobs = np.concatenate((self.logprobs, logprobs[None, :]))
        self.values = np.concatenate((self.values, values[None, :]))
        self.rewards = np.concatenate((self.rewards, rewards[None, :]))
        self.terminated = np.concatenate((self.terminated, terminated[None, :]))

    def get_minibatches(
        self, next_value: Tensor, next_terminated: Tensor, gamma: float, gae_lambda: float
    ) -> list[ReplayMinibatch]:
        """Returns a list of minibatches, the union being `batches_per_learning_phase` copies
        of the entire replay memory."""
        obs, actions, logprobs, values, rewards, terminated = (
            t.tensor(x, device=device, dtype=t.float32)
            for x in [self.obs, self.actions, self.logprobs, self.values, self.rewards, self.terminated]
        )

        advantages = compute_advantages(next_value, next_terminated, rewards, values, terminated, gamma, gae_lambda)
        returns = advantages + values

        minibatches = []
        for _ in range(self.batches_per_learning_phase):
            for indices in get_minibatch_indices(self.rng, self.batch_size, self.minibatch_size):
                minibatches.append(
                    ReplayMinibatch(
                        *[
                            data.flatten(0, 1)[indices]
                            for data in [obs, actions, logprobs, advantages, returns, terminated]
                        ]
                    )
                )

        self.reset()
        return minibatches


class PPOAgent:
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv, actor: nn.Module, critic: nn.Module, memory: ReplayMemory):
        super().__init__()
        self.envs = envs
        self.actor = actor
        self.critic = critic
        self.memory = memory

        self.step = 0  # Tracking number of steps taken (across all environments)
        self.next_obs = t.tensor(envs.reset()[0], device=device, dtype=t.float)
        self.next_terminated = t.zeros(envs.num_envs, device=device, dtype=t.bool)

    def play_step(self) -> list[dict]:
        """Carries out a single interaction step between agent and env, adding results to memory.

        NOTE: if you use action masking, apply the env's info["action_mask"] to `logits`
        (set masked logits to -inf) before building the Categorical here AND in
        PPOTrainer.compute_ppo_objective, so the sampled action and its logprob agree.
        """
        obs = self.next_obs
        terminated = self.next_terminated

        with t.inference_mode():
            logits = self.actor(obs)
            dist = t.distributions.categorical.Categorical(logits=logits)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            values = self.critic(obs).flatten()

        next_obs, rewards, next_terminated, next_truncated, infos = self.envs.step(actions.cpu().numpy())
        self.memory.add(
            obs.cpu().numpy(),
            actions.cpu().numpy(),
            logprobs.cpu().numpy(),
            values.cpu().numpy(),
            rewards,
            terminated.cpu().numpy(),
        )

        self.next_obs = t.tensor(next_obs, device=device, dtype=t.float)
        self.next_terminated = t.tensor(next_terminated, device=device, dtype=t.float)
        self.step += self.envs.num_envs
        return infos

    def get_minibatches(self, gamma: float, gae_lambda: float) -> list[ReplayMinibatch]:
        """Gets minibatches from the replay memory, and resets the memory."""
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        minibatches = self.memory.get_minibatches(next_value, self.next_terminated, gamma, gae_lambda)
        self.memory.reset()
        return minibatches


def calc_clipped_surrogate_objective(
    dist: Categorical,
    mb_action: Int[Tensor, "minibatch_size"],
    mb_advantages: Float[Tensor, "minibatch_size"],
    mb_logprobs: Float[Tensor, "minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    """Return the clipped surrogate objective, suitable for maximisation with gradient ascent."""
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape
    cur_logprobs = dist.log_prob(mb_action)
    r = t.exp(cur_logprobs - mb_logprobs)
    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)
    return t.min(r * mb_advantages, t.clip(r, 1 - clip_coef, 1 + clip_coef) * mb_advantages).mean()


def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"],
    mb_returns: Float[Tensor, "minibatch_size"],
    vf_coef: float,
) -> Float[Tensor, ""]:
    """Compute the value function portion of the loss function."""
    assert values.shape == mb_returns.shape
    return vf_coef * (values - mb_returns).pow(2).mean()


def calc_entropy_bonus(dist: Categorical, ent_coef: float):
    """Return the entropy bonus term, suitable for gradient ascent."""
    return ent_coef * dist.entropy().mean()


class PPOScheduler:
    def __init__(self, optimizer: Optimizer, initial_lr: float, end_lr: float, total_phases: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_phases = total_phases
        self.n_step_calls = 0

    def step(self):
        """Linear learning rate decay so that after `total_phases` calls, lr == end_lr."""
        self.n_step_calls += 1
        frac = min(1, self.n_step_calls / self.total_phases)
        lr = (1 - frac) * self.initial_lr + frac * self.end_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def make_optimizer(
    actor: nn.Module, critic: nn.Module, total_phases: int, initial_lr: float, end_lr: float = 0.0
) -> tuple[optim.Adam, PPOScheduler]:
    """Return an appropriately configured Adam with its attached scheduler."""
    optimizer = optim.AdamW(
        itertools.chain(actor.parameters(), critic.parameters()),
        lr=initial_lr,
        eps=1e-5,
        maximize=True,
    )
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_phases)
    return optimizer, scheduler


class PPOTrainer:
    def __init__(self, args: PPOArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.run_name = f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(idx=idx, run_name=self.run_name, **args.__dict__) for idx in range(args.num_envs)]
        )

        self.num_envs = self.envs.num_envs
        self.action_shape = self.envs.single_action_space.shape
        self.obs_shape = self.envs.single_observation_space.shape

        self.memory = ReplayMemory(
            self.num_envs,
            self.obs_shape,
            self.action_shape,
            args.batch_size,
            args.minibatch_size,
            args.batches_per_learning_phase,
            args.seed,
        )

        self.actor, self.critic = get_actor_and_critic(self.envs)
        self.optimizer, self.scheduler = make_optimizer(self.actor, self.critic, args.total_training_steps, args.lr)

        self.agent = PPOAgent(self.envs, self.actor, self.critic, self.memory)

    def rollout_phase(self) -> dict | None:
        """Populates the memory with a new set of experiences via self.agent.play_step, and
        returns a dict of data for the progress bar postfix."""
        data = None
        t0 = time.time()

        for step in range(self.args.num_steps_per_rollout):
            infos = self.agent.play_step()

            new_data = get_episode_data_from_infos(infos)
            if new_data is not None:
                data = new_data
                if self.args.use_wandb:
                    wandb.log(new_data, step=self.agent.step)

        if self.args.use_wandb:
            wandb.log(
                {"SPS": (self.args.num_steps_per_rollout * self.num_envs) / (time.time() - t0)}, step=self.agent.step
            )

        return data

    def learning_phase(self) -> None:
        """Generates minibatches, computes the objective, steps the optimizer & scheduler."""
        minibatches = self.agent.get_minibatches(self.args.gamma, self.args.gae_lambda)
        for minibatch in minibatches:
            obj = self.compute_ppo_objective(minibatch)
            obj.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), self.args.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()

    def compute_ppo_objective(self, minibatch: ReplayMinibatch) -> Float[Tensor, ""]:
        """Handles learning phase for a single minibatch. Returns objective to be maximized."""
        logits = self.actor(minibatch.obs)
        values = self.critic(minibatch.obs).flatten()

        dist = Categorical(logits=logits)
        clipped_surrogate_objective = calc_clipped_surrogate_objective(
            dist, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef
        )
        entropy_bonus = calc_entropy_bonus(dist, self.args.ent_coef)
        value_loss = calc_value_function_loss(values, minibatch.returns, self.args.vf_coef)

        if self.args.use_wandb:
            with t.inference_mode():
                newlogprob = dist.log_prob(minibatch.actions)
                logratio = newlogprob - minibatch.logprobs
                ratio = logratio.exp()
                approx_kl = (ratio - 1 - logratio).mean().item()
                clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
            wandb.log(
                dict(
                    total_steps=self.agent.step,
                    values=values.mean().item(),
                    lr=self.scheduler.optimizer.param_groups[0]["lr"],
                    value_loss=value_loss.item(),
                    clipped_surrogate_objective=clipped_surrogate_objective.item(),
                    entropy=entropy_bonus.item(),
                    approx_kl=approx_kl,
                    clipfrac=np.mean(clipfracs),
                ),
                step=self.agent.step,
            )

        return clipped_surrogate_objective + entropy_bonus - value_loss

    def train(self) -> None:
        run = None
        if self.args.use_wandb:
            run = wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                monitor_gym=self.args.video_log_freq is not None,
            )
            wandb.watch([self.actor, self.critic], log="all", log_freq=50)

        try:
            pbar = tqdm(range(self.args.total_phases))
            last_logged_time = time.time()

            for phase in pbar:
                data = self.rollout_phase()
                if data is not None and time.time() - last_logged_time > 0.5:
                    last_logged_time = time.time()
                    pbar.set_postfix(phase=phase, **data)

                self.learning_phase()

        except KeyboardInterrupt:
            print("Training interrupted, shutting down cleanly.")
        finally:
            self.envs.close()
            if run:
                try:
                    run.finish()
                except (Exception, KeyboardInterrupt):
                    wandb.teardown()
