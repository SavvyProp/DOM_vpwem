"""InterceptFast with the middle third of its overhead cover removed."""

from dom_vpwem.custom_envs.intercept_fast_cover import InterceptFastCover


class InterceptFastCover2(InterceptFastCover):
    """Split the cover along +Y, the ball's main direction of travel.

    Keep the original X/Z bounds and the outer thirds of its Y extent:
    [-0.50, -0.3666667] and [-0.2333333, -0.10] metres. The middle
    13.33 cm is open. Both sections have matching visual and collision boxes.
    A higher fixed arm pose frames both launch and the opening from above.
    The inherited wrapper holds the arm for the first five control steps.
    """

    # Frame the launch range and the gap with one fixed 90-degree wrist view.
    # Offline URDF calibration puts the camera near (-0.45, -0.45, 0.85) m:
    # high enough to see into the gap over the first section. The base variant's
    # reset hook uses these constants, never the sampled ball position.
    INITIAL_LOOK_TARGET = (-0.20, -0.575, 0.03)
    INITIAL_QPOS = (-1.31366, -0.07368, -0.04344, -1.00494, 0.27482, 1.15604, 0.83327, 0.04, 0.04)

    def _load_cover(self) -> None:
        x, y, z = self.COVER_CENTER
        width, length, height = self.COVER_SIZE
        section_length = length / 3
        self.covers = tuple(
            self._build_cover(
                name=f"cover_{index}",
                center=(x, y + direction * section_length, z),
                size=(width, section_length, height),
            )
            for index, direction in enumerate((-1, 1))
        )
