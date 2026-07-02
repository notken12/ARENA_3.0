"""
Wordle gymnasium environment.

Implemented here: the full game *mechanics* — word-list loading, answer sampling,
feedback computation (with correct duplicate-letter handling), guess-history tracking,
termination, and the exact action mask / candidate filtering.

Left as stubs for you to implement (the project-specific design choices):
    - WordleEnv.encode_observation  -> your belief-state encoding
    - WordleEnv.observation_space   -> shape implied by that encoding
    - WordleEnv.compute_reward      -> your reward shaping
"""

from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
from jaxtyping import Bool, Float, Int

Arr = np.ndarray

WORD_LEN = 5
ALPHABET_SIZE = 26
MAX_GUESSES = 6

# Feedback tile values.
GRAY = 0  # letter not in word (accounting for counts)
YELLOW = 1  # letter in word, wrong position
GREEN = 2  # letter in word, correct position

# Official Wordle word lists (newline-separated, one 5-letter word per line).
#   answers  ~2,315 words  (the possible solutions)
#   allowed ~12,972 words  (every word accepted as a guess; a superset of answers)
# Drop these two files in DATA_DIR. They are widely mirrored, e.g. the lists used by
# the "wordle-solver" projects on GitHub.
DATA_DIR = Path(__file__).resolve().parent / "wordle_data"
ANSWERS_PATH = DATA_DIR / "wordle_answers.txt"
ALLOWED_PATH = DATA_DIR / "wordle_allowed.txt"


def load_word_list(path: Path) -> list[str]:
    """Load a newline-separated list of lowercase 5-letter words."""
    if not path.exists():
        raise FileNotFoundError(
            f"Word list not found at {path}. Download the official Wordle answer and "
            f"allowed-guess lists and place them at {ANSWERS_PATH} and {ALLOWED_PATH}."
        )
    words = [w.strip().lower() for w in path.read_text().splitlines() if w.strip()]
    assert all(len(w) == WORD_LEN and w.isalpha() for w in words), f"malformed word in {path}"
    return words


def compute_feedback(guess: str, answer: str) -> Int[Arr, " WORD_LEN"]:
    """
    Return the per-position feedback for `guess` against `answer` using Wordle's rules,
    including correct duplicate-letter accounting (two-pass: greens first, then yellows
    drawn from the remaining letter pool).
    """
    feedback = np.full(WORD_LEN, GRAY, dtype=np.int8)
    remaining = Counter(answer)

    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            feedback[i] = GREEN
            remaining[g] -= 1

    for i, g in enumerate(guess):
        if feedback[i] == GREEN:
            continue
        if remaining[g] > 0:
            feedback[i] = YELLOW
            remaining[g] -= 1

    return feedback


def is_consistent(candidate: str, guess: str, feedback: Int[Arr, " WORD_LEN"]) -> bool:
    """A candidate answer is consistent with (guess, feedback) iff guessing it would have
    produced exactly that feedback."""
    return np.array_equal(compute_feedback(guess, candidate), feedback)


class WordleEnv(gym.Env):
    """
    Gymnasium Wordle environment.

    Action space: Discrete(len(allowed_guesses)) — the index of the word to guess.
    Observation:  your belief-state encoding (see `encode_observation`, currently a stub).

    Episode: up to MAX_GUESSES guesses; terminates on a correct guess or when guesses
    run out. The hidden answer is sampled uniformly from `answers` on each reset.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        answers: list[str] | None = None,
        allowed_guesses: list[str] | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.answers = answers if answers is not None else load_word_list(ANSWERS_PATH)
        self.allowed_guesses = allowed_guesses if allowed_guesses is not None else load_word_list(ALLOWED_PATH)

        self.action_space = gym.spaces.Discrete(len(self.allowed_guesses))

        # TODO: define once you've chosen your belief-state encoding (see encode_observation).
        # The belief-state design we discussed flattens to:
        #   green [5,26] + position_excluded [5,26] + present [26] + absent [26]
        #   + letter_counts [26] + guesses_remaining [1]  ->  length 5*26*2 + 26*3 + 1 = 339
        #   self.observation_space = gym.spaces.Box(0.0, HIGH, shape=(OBS_LEN,), dtype=np.float32)
        self.observation_space = None  # TODO

        # Episode state (populated in reset()).
        self.answer: str = ""
        self.history: list[tuple[str, Int[Arr, " WORD_LEN"]]] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.answer = self.answers[self.np_random.integers(len(self.answers))]
        self.history = []
        return self.encode_observation(), self._info()

    def step(self, action: int):
        guess = self.allowed_guesses[action]
        feedback = compute_feedback(guess, self.answer)
        self.history.append((guess, feedback))

        solved = guess == self.answer
        out_of_guesses = len(self.history) >= MAX_GUESSES
        terminated = solved or out_of_guesses

        reward = self.compute_reward(guess, feedback, solved, terminated)
        return self.encode_observation(), reward, terminated, False, self._info(solved=solved)

    def _info(self, solved: bool = False) -> dict:
        return {
            "num_guesses": len(self.history),
            "solved": solved,
            "action_mask": self.get_action_mask(),
        }

    # --- game-logic helpers (implemented) ------------------------------------------------

    def consistent_answers(self) -> list[str]:
        """The set of possible answers still consistent with all feedback so far.
        Useful for an information-gain reward: |C_t| is len(consistent_answers())."""
        candidates = self.answers
        for guess, feedback in self.history:
            candidates = [w for w in candidates if is_consistent(w, guess, feedback)]
        return candidates

    def get_action_mask(self) -> Bool[Arr, " num_allowed"]:
        """Boolean mask over `allowed_guesses`: True where the word is consistent with all
        feedback so far (i.e. hard-mode legal). Only paid when you call it.

        NOTE: O(num_allowed * history_len * WORD_LEN) in pure Python. If it becomes a
        bottleneck, precompute an int-encoded [num_allowed, WORD_LEN] matrix of the guess
        list and vectorize `is_consistent` over it with numpy.
        """
        mask = np.ones(len(self.allowed_guesses), dtype=bool)
        for i, word in enumerate(self.allowed_guesses):
            for guess, feedback in self.history:
                if not is_consistent(word, guess, feedback):
                    mask[i] = False
                    break
        return mask

    def render(self):
        return None

    # --- STUBS: your project-specific design choices -------------------------------------

    def encode_observation(self) -> Float[Arr, " *obs_shape"]:
        """
        TODO: turn `self.history` into your belief-state observation (a float32 array
        matching self.observation_space). The belief state we discussed, per letter/position:
            green[p, l], position_excluded[p, l], present[l], absent[l], letter_counts[l],
            guesses_remaining.
        Compute this from the *actual* feedback in self.history (not a lossy tensor) so
        duplicate-letter constraints stay exact.
        """
        raise NotImplementedError

    def compute_reward(self, guess: str, feedback: Int[Arr, " WORD_LEN"], solved: bool, terminated: bool) -> float:
        """
        TODO: your reward shaping. Options we discussed:
            - Sparse: +1 on `solved`, else 0 (rely on gamma<1 to reward faster solves).
            - Dense information gain: alpha * (log|C_{t-1}| - log|C_t|) + solve bonus,
              where |C_t| = len(self.consistent_answers()).
        """
        raise NotImplementedError


gym.envs.registration.register(id="Wordle-v0", entry_point=WordleEnv, max_episode_steps=MAX_GUESSES)


if __name__ == "__main__":
    # Sanity-check the trickiest mechanic (duplicate-letter feedback) without needing the
    # full word lists. G=green, Y=yellow, _=gray.
    def fb(guess, answer):
        return "".join("_YG"[v] for v in compute_feedback(guess, answer))

    assert fb("crane", "crane") == "GGGGG"
    assert fb("aaaaa", "crane") == "__G__"  # only the aligned 'a' is green; the other 'a's are gray
    assert fb("eerie", "crane") == "__Y_G"  # answer has one 'e' (green at end); extra 'e's gray; 'r' yellow
    assert fb("babes", "abbey") == "YYGG_"  # duplicate 'b' handled across positions (one green, one yellow)
    assert fb("llama", "hello") == "YY___"  # answer has two 'l's, both guessed 'l's yellow; 'a' gray
    print("compute_feedback duplicate-letter tests passed!")
