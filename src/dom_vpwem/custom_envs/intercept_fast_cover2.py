"""InterceptFast with the middle third of its overhead cover removed."""

from dom_vpwem.custom_envs.intercept_fast_cover import InterceptFastCover


class InterceptFastCover2(InterceptFastCover):
    """Split the cover along +Y, the ball's main direction of travel.

    Keep the original X/Z bounds and the outer thirds of its Y extent:
    [-0.50, -0.3666667] and [-0.2333333, -0.10] metres. The middle
    13.33 cm is open. Both sections are opaque and collisionless.
    Visibility through the gap depends on the policy camera's viewpoint.
    """

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
