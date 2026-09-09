"""InterceptFast with a fixed opaque cover above the rolling ball's path."""

import sapien
from mani_skill.utils.structs import Actor
from mikasa_robo_suite.vla.memory_envs.intercept_vla import InterceptFastVLAEnv


class InterceptFastCover(InterceptFastVLAEnv):
    """Keep InterceptFast physics and task logic; add visual occlusion only.

    All measurements are metres in the task's world frame (tabletop z=0).
    Bounds: x=[-0.40, 0.00], y=[-0.50, -0.10], z=[0.20, 0.30].
    The fixed and wrist cameras see the cover from their respective viewpoints;
    the wrist camera can see beneath it. It has no collision geometry.
    """

    COVER_CENTER = (-0.20, -0.30, 0.25)
    COVER_SIZE = (0.40, 0.40, 0.10)
    COVER_COLOR = (0.25, 0.25, 0.25, 1.0)

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
        """Build one opaque, collisionless section; dimensions are full sizes."""
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[dimension / 2 for dimension in size],
            material=sapien.render.RenderMaterial(base_color=self.COVER_COLOR),
        )
        builder.initial_pose = sapien.Pose(p=center)
        # A static visual actor stays in place across steps and resets. Omitting
        # collision shapes lets the robot and ball pass through the cover.
        return builder.build_static(name=name)
