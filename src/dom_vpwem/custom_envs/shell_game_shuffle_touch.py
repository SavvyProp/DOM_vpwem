"""Repository-owned shell-game shuffle-and-touch task."""

from mikasa_robo_suite.vla.memory_envs.shell_game_shuffle_touch_vla import (
    ShellGameShuffleTouchVLAEnv,
)


class ShellGameShuffleTouch(ShellGameShuffleTouchVLAEnv):
    """Start with MIKASA's short task and customize its settings here.

    Observe the ball during the cue, track its cup through the shuffle,
    then touch that cup. Scene construction, rewards, and observations are
    inherited. The task registry supplies the 60-step horizon and the base
    task's wrapper, which freezes robot actions during cue and shuffle.
    """

    # Inclusive ranges, in control steps except for the number of swaps.
    CUE_PHASE_STEPS = [1, 5]
    SHUFFLE_PHASE_STEPS = [20, 35]
    NUM_SWAPS = [2, 4]
    SWAP_ARC_HEIGHT = 0.06  # metres
