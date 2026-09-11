"""InterceptFast with a fixed opaque cover above the rolling ball's path."""

from math import pi

import sapien
import torch
from mani_skill.utils.structs import Actor
from mikasa_robo_suite.vla.memory_envs.intercept_vla import InterceptFastVLAEnv
from mikasa_robo_suite.vla.utils.wrappers import CurriculumPhaseNoopActionWrapper


class InterceptCueNoopActionWrapper(CurriculumPhaseNoopActionWrapper):
    """Hold robot actions at zero while the ball begins its launch."""

    def _get_noop_mask(self):
        base = self.env.unwrapped
        return base.elapsed_steps < base.CUE_STEPS


class InterceptFastCover(InterceptFastVLAEnv):
    """Add visual occlusion and a fixed robot pose looking toward ball launch.

    All measurements are metres in the task's world frame (tabletop z=0).
    Bounds: x=[-0.40, 0.00], y=[-0.50, -0.10], z=[0.20, 0.30].
    The fixed and wrist cameras see the cover from their respective viewpoints;
    the wrist camera can see beneath it. Matching static collision geometry
    blocks the robot while leaving the rolling ball 0.16 m of clearance.
    The registered wrapper suppresses the first five robot actions while the
    ball moves normally, then releases the arm on the sixth action.
    """

    CURRICULUM_WRAPPER = InterceptCueNoopActionWrapper
    CUE_STEPS = 5
    COVER_CENTER = (-0.20, -0.30, 0.25)
    COVER_SIZE = (0.40, 0.40, 0.10)
    COVER_COLOR = (0.25, 0.25, 0.25, 1.0)

    # Mean of the upstream launch distribution, for documenting/calibrating
    # the fixed pose. Never aim at the sampled ball position during reset.
    INITIAL_LOOK_TARGET = (-0.20, -0.775, 0.04)
    # Turn the shoulder toward -Y and tilt the wrist toward the launch area.
    # Rounded, offline-calibrated joint angles; the wrist camera keeps its
    # original mount and 90-degree field of view and moves with the real arm.
    INITIAL_QPOS = (-1.10, 0.0, 0.0, -2 * pi / 3, 0.0, 2.60, pi / 4, 0.04, 0.04)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        super()._initialize_episode(env_idx, options)
        # Replace the inherited noisy arm pose after ball/goal initialization.
        # ManiSkill applies these rows only to the environments being reset,
        # then resets controller targets after this hook returns.
        qpos = torch.tensor(self.INITIAL_QPOS, dtype=torch.float32, device=self.device)
        self.agent.reset(qpos.expand(len(env_idx), -1))

    def _load_scene(self, options: dict) -> None:
        super()._load_scene(options)
        self._load_cover()

    def _load_cover(self) -> None:
        self.cover = self._build_cover("cover", self.COVER_CENTER, self.COVER_SIZE)

    def _build_cover(
        self,
        name: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
    ) -> Actor:
        """Build one solid opaque section; dimensions are full sizes."""
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[dimension / 2 for dimension in size])
        builder.add_box_visual(
            half_size=[dimension / 2 for dimension in size],
            material=sapien.render.RenderMaterial(base_color=self.COVER_COLOR),
        )
        builder.initial_pose = sapien.Pose(p=center)
        # Keep the collider aligned with the visual. The ball travels below
        # its z=0.20 m underside; no support walls obstruct the rolling path.
        return builder.build_static(name=name)
