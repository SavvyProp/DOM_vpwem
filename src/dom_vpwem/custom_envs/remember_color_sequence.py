"""Remember the n-th color from the end of a variable-length cue sequence."""

from numbers import Integral

import numpy as np
import sapien
import torch
from mani_skill.utils.structs.pose import Pose
from mikasa_robo_suite.vla.memory_envs.remember_color_vla import RememberColor3LongVLAEnv


class RememberColorSequence3Long(RememberColor3LongVLAEnv):
    """Show one color at a time, with blank gaps, then offer three choices.

    The last item is position 1 from the end. Colors are sampled independently
    with replacement, so repetitions are allowed. Sequence length and the
    final blank delay are sampled independently for each episode.

    All durations are control steps. Defaults use at most 100 sequence steps
    plus 450 delay steps, leaving at least 50 of the registered 600 steps for
    answering. Increase the registered/configured horizon if extending this.
    """

    TARGET_FROM_END = 2
    SEQUENCE_LENGTH_RANGE = (3, 7)
    COLOR_STEPS = 10
    GAP_STEPS = 5
    EMPTY_PHASE_STEPS = (50, 450)
    LANGUAGE_INSTRUCTION = (
        "Observe the sequence of colors, wait, then touch the cube matching "
        "the second-to-last color."
    )

    def __init__(self, *args, target_from_end: int | None = None, **kwargs):
        self.target_from_end = self.TARGET_FROM_END if target_from_end is None else target_from_end
        self._validate_sequence_settings()
        if self.target_from_end != 2:
            self.LANGUAGE_INSTRUCTION = (
                "Observe the sequence of colors, wait, then touch the cube matching "
                f"color number {self.target_from_end} counting backward from the end "
                "(the last color is number 1)."
            )
        super().__init__(*args, **kwargs)

    def _validate_sequence_settings(self) -> None:
        for name, value in (
            ("target_from_end", self.target_from_end),
            ("COLOR_STEPS", self.COLOR_STEPS),
            ("GAP_STEPS", self.GAP_STEPS),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("SEQUENCE_LENGTH_RANGE", "EMPTY_PHASE_STEPS"):
            bounds = getattr(self, name)
            if (
                not isinstance(bounds, (tuple, list))
                or len(bounds) != 2
                or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"{name} must contain two positive integers in ascending order")
        if self.SEQUENCE_LENGTH_RANGE[0] < self.target_from_end:
            raise ValueError("The minimum sequence length must be at least target_from_end")

    def _load_scene(self, options: dict) -> None:
        super()._load_scene(options)
        self._allocate_episode_buffers()

    def _allocate_episode_buffers(self) -> None:
        self.sequence_colors = torch.zeros(
            (self.num_envs, self.SEQUENCE_LENGTH_RANGE[1]), dtype=torch.int64, device=self.device
        )
        self.sequence_lengths = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.cue_steps_per_env = torch.zeros_like(self.sequence_lengths)
        self.empty_steps_per_env = torch.zeros_like(self.sequence_lengths)
        self.true_color_indices = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)
        self.oracle_info = self.true_color_indices
        self._choices_revealed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.initial_poses = {
            key: torch.zeros((self.num_envs, 3), device=self.device) for key in self.color_dict
        }
        self.center_pose = torch.tensor([0, 0, self.CUBE_HALFSIZE], device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        # The upstream initializer rebuilds some buffers using the reset batch
        # size. Keep full-size buffers and update only env_idx for PPO autoresets.
        rng = self._batched_episode_rng[env_idx]
        b = len(env_idx)
        lengths = rng.randint(self.SEQUENCE_LENGTH_RANGE[0], self.SEQUENCE_LENGTH_RANGE[1] + 1)
        colors = rng.randint(self.COLORS, size=self.SEQUENCE_LENGTH_RANGE[1])
        delays = rng.randint(self.EMPTY_PHASE_STEPS[0], self.EMPTY_PHASE_STEPS[1] + 1)
        self.sequence_lengths[env_idx] = torch.as_tensor(lengths, device=self.device)
        self.sequence_colors[env_idx] = torch.as_tensor(colors, device=self.device)
        self.true_color_indices[env_idx] = torch.as_tensor(
            colors[np.arange(b), lengths - self.target_from_end],
            dtype=torch.uint8,
            device=self.device,
        )
        self.cue_steps_per_env[env_idx] = (
            self.sequence_lengths[env_idx] * (self.COLOR_STEPS + self.GAP_STEPS) - self.GAP_STEPS
        )
        self.empty_steps_per_env[env_idx] = torch.as_tensor(delays, device=self.device)
        self._choices_revealed[env_idx] = False

        # Randomize color-to-slot assignments independently of the cue sequence.
        # Slot spacing and bounded jitter keep the answer cubes well separated.
        positions = np.zeros((b, self.COLORS, 3), dtype=np.float32)
        positions[:, :, 1] = np.linspace(-0.2, 0.2, self.COLORS)
        positions[:, :, :2] += rng.uniform(-0.01, 0.01, size=(self.COLORS, 2))
        positions[:, :, 2] = self.CUBE_HALFSIZE
        permutations = rng.permutation(self.COLORS)
        positions = positions[np.arange(b)[:, None], permutations]

        qpos = np.tile([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04], (b, 1))
        qpos[:, :-2] += rng.normal(0, self.robot_init_qpos_noise, size=7)
        self.table_scene.initialize(env_idx)
        self.agent.reset(qpos)
        self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
        for key, cube in self.cubes.items():
            xyz = torch.as_tensor(positions[:, key], device=self.device)
            self.initial_poses[key][env_idx] = xyz
            cube.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))

        self.task_cue = None
        self.reward_dict = None
        if getattr(self, "_prev_action", None) is not None:
            self._prev_action[env_idx] = 0

    def _sequence_phase(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the visible cue color (-1 for blank) and the answer-phase mask."""
        elapsed = self.elapsed_steps.to(torch.int64)
        period = self.COLOR_STEPS + self.GAP_STEPS
        item = (elapsed // period).clamp(max=self.SEQUENCE_LENGTH_RANGE[1] - 1)
        color = self.sequence_colors.gather(1, item[:, None]).squeeze(1)
        visible = (elapsed < self.cue_steps_per_env) & (elapsed % period < self.COLOR_STEPS)
        color = torch.where(visible, color, -1)
        answer = elapsed >= self.cue_steps_per_env + self.empty_steps_per_env
        return color, answer

    def evaluate(self) -> dict:
        visible_color, self.manip_mask = self._sequence_phase()
        hidden = ~self.manip_mask
        appeared = self.manip_mask & ~self._choices_revealed
        locked = hidden | appeared
        self.original_poses = {key: cube.pose.raw_pose.clone() for key, cube in self.cubes.items()}
        for key, cube in self.cubes.items():
            pose = cube.pose.raw_pose.clone()
            # Keep hidden dynamic cubes apart so they do not collide with one
            # another while outside the camera's view.
            pose[hidden, :3] = self.initial_poses[key][hidden]
            pose[hidden, 2] = 1000
            pose[visible_color == key, :3] = self.center_pose
            pose[appeared, :3] = self.initial_poses[key][appeared]
            pose[locked, 3:] = torch.tensor([1, 0, 0, 0], device=self.device, dtype=pose.dtype)
            cube.pose = pose
            # Repeated colors must disappear during gaps. Reset both pose and
            # velocity while presenting cues, then release the answer cubes.
            linear, angular = cube.linear_velocity.clone(), cube.angular_velocity.clone()
            linear[locked] = 0
            angular[locked] = 0
            cube.set_linear_velocity(linear)
            cube.set_angular_velocity(angular)
        self._choices_revealed |= self.manip_mask

        self.masks = {key: (self.true_color_indices == key)[:, None] for key in self.cubes}
        self.obj_to_goal_pos = torch.zeros_like(self.cubes[0].pose.p)
        for key, cube in self.cubes.items():
            self.obj_to_goal_pos += (cube.pose.p - self.agent.tcp.pose.p) * self.masks[key]
        placed = torch.linalg.vector_norm(self.obj_to_goal_pos, dim=1) <= self.GOAL_THRESH
        placed &= self.manip_mask
        static = self.agent.is_static(0.2)
        return {
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "is_obj_placed": placed,
            "is_robot_static": static,
            "success": placed & static,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: dict) -> dict:
        obs = super()._get_obs_extra(info)
        if self.obs_mode in ("state", "state_dict"):
            # Privileged timing helps a feed-forward PPO oracle wait to answer.
            # RGB students receive neither this timing nor the target label.
            obs["answer_phase"] = self.manip_mask[:, None].float()
            obs["steps_until_choices"] = (
                (self.cue_steps_per_env + self.empty_steps_per_env - self.elapsed_steps)
                .clamp(min=0)[:, None]
                .float()
            )
        return obs

    def compute_dense_reward(self, obs, action, info: dict) -> torch.Tensor:
        reward = super().compute_dense_reward(obs, action, info)
        return torch.where(self.manip_mask, reward, 0.0)
