"""Intercept cue timing and real CPU collisions without a Vulkan renderer."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(params=["InterceptFastCover", "InterceptFastCover2"])
def intercept_class(request):
    pytest.importorskip("mikasa_robo_suite")
    from dom_vpwem.custom_envs.intercept_fast_cover import InterceptFastCover
    from dom_vpwem.custom_envs.intercept_fast_cover2 import InterceptFastCover2

    return {cls.__name__: cls for cls in (InterceptFastCover, InterceptFastCover2)}[request.param]


@pytest.mark.parametrize("numpy_action", [False, True])
@pytest.mark.parametrize("flat_action", [False, True])
def test_first_five_actions_are_suppressed_and_partial_reset_restarts_cue(
    intercept_class, numpy_action, flat_action
):
    import gymnasium as gym

    class RawEnv(gym.Env):
        CUE_STEPS = intercept_class.CUE_STEPS

        def reset(self, *, seed=None, options=None):
            indices = (options or {}).get("env_idx", [0, 1])
            if not hasattr(self, "elapsed_steps"):
                self.elapsed_steps = torch.zeros(2, dtype=torch.int64)
                self.ball_y = torch.zeros(2)
            self.elapsed_steps[indices] = 0
            self.ball_y[indices] = -0.8
            return {}, {"elapsed_steps": self.elapsed_steps.clone()}

        def step(self, action):
            self.executed = torch.as_tensor(action).expand(2, 7).clone()
            self.elapsed_steps += 1
            self.ball_y += 0.05
            return {}, 0.0, False, False, {"elapsed_steps": self.elapsed_steps.clone()}

    raw = RawEnv()
    env = intercept_class.CURRICULUM_WRAPPER(raw)
    env.reset()
    action = torch.full((7,) if flat_action else (2, 7), 0.75)
    if numpy_action:
        action = action.numpy()
    assert raw.CUE_STEPS == 5
    for step in range(6):
        env.step(action)
        torch.testing.assert_close(
            raw.executed, torch.full((2, 7), 0.0 if step < 5 else 0.75)
        )
        torch.testing.assert_close(raw.ball_y, torch.full((2,), -0.8 + 0.05 * (step + 1)))
        assert torch.all(torch.as_tensor(action) == 0.75)
    env.reset(options={"env_idx": [1]})
    env.step(action)
    torch.testing.assert_close(raw.executed[0], torch.full((7,), 0.75))
    torch.testing.assert_close(raw.executed[1], torch.zeros(7))
    assert raw.elapsed_steps.tolist() == [7, 1]


@pytest.fixture
def roll_ball(intercept_class, monkeypatch):
    import sapien
    from mani_skill.envs.sapien_env import BaseEnv

    config = object.__new__(intercept_class)._default_sim_config
    BaseEnv._set_scene_config(SimpleNamespace(sim_config=config))
    # Rendering is omitted, but production _load_cover/_build_cover create
    # the actual collision shapes, sizes, poses, and static bodies below.
    monkeypatch.setattr(sapien.render, "RenderMaterial", lambda **kwargs: None)

    class CollisionBuilder:
        def __init__(self, scene):
            self.builder = scene.create_actor_builder()

        def __getattr__(self, name):
            return getattr(self.builder, name)

        @property
        def initial_pose(self):
            return self.builder.initial_pose

        @initial_pose.setter
        def initial_pose(self, value):
            self.builder.initial_pose = value

        def add_box_visual(self, **kwargs):
            pass

    def run(launch, *, covers):
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        scene.set_timestep(1 / config.sim_freq)
        builder = scene.create_actor_builder()
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
            half_size=[2.418 / 2, 1.209 / 2, 0.9196429 / 2],
        )
        builder.initial_pose = sapien.Pose(
            p=[-0.12, 0, -0.9196429], q=[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        )
        builder.build_kinematic(name="table")
        if covers:
            env = object.__new__(intercept_class)
            env.scene = SimpleNamespace(create_actor_builder=lambda: CollisionBuilder(scene))
            env._load_cover()
        x, y, z, vx, vy = launch
        builder = scene.create_actor_builder()
        builder.add_sphere_collision(radius=intercept_class.BALL_RADIUS)
        builder.initial_pose = sapien.Pose(p=[x, y, z])
        ball = builder.build(name="ball")
        body = ball.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        body.linear_velocity = [vx, vy, 0]
        trajectory = [ball.pose.p.copy()]
        contacts = 0
        try:
            for _ in range(2 * config.sim_freq):
                scene.step()
                trajectory.append(ball.pose.p.copy())
                for contact in scene.physx_system.get_contacts():
                    names = {component.entity.name for component in contact.bodies}
                    if "ball" in names and any(name.startswith("cover") for name in names):
                        contacts += 1
            return np.asarray(trajectory), contacts
        finally:
            scene.clear()

    return run


def test_real_cover_colliders_leave_ball_launches_unchanged(intercept_class, roll_ball):
    radius = intercept_class.BALL_RADIUS
    for x, y, vx, vy in itertools.product((-0.35, -0.05), (-0.90, -0.65), (0, 0.05), (0.75, 1)):
        launch = (x, y, 2 * radius, vx, vy)
        baseline, _ = roll_ball(launch, covers=False)
        covered, contacts = roll_ball(launch, covers=True)
        np.testing.assert_allclose(covered, baseline, rtol=0, atol=1e-6)
        assert contacts == 0
        assert covered[-1, 1] - radius > -0.10
    # Positive control: a sphere falling onto a solid section must be stopped
    # by its top. This fails if the production builder adds no collider.
    section_y = -0.30 if intercept_class.__name__ == "InterceptFastCover" else -13 / 30
    probe, contacts = roll_ball((-0.2, section_y, 0.45, 0, 0), covers=True)
    assert contacts > 0
    assert probe[-1, 2] == pytest.approx(0.30 + radius, abs=1e-4)
