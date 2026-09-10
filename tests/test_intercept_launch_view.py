"""Fixed reset pose and launch visibility for both Intercept cover variants."""

import itertools
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(params=["InterceptFastCover", "InterceptFastCover2"])
def cover_class(request):
    pytest.importorskip("mikasa_robo_suite")
    from dom_vpwem.custom_envs.intercept_fast_cover import InterceptFastCover
    from dom_vpwem.custom_envs.intercept_fast_cover2 import InterceptFastCover2

    return {cls.__name__: cls for cls in (InterceptFastCover, InterceptFastCover2)}[request.param]


def test_reset_pose_is_fixed_across_launches_and_respects_partial_resets(cover_class):
    # Exercise the actual inherited launch sampling and the local reset hook.
    # Tensor actors emulate ManiSkill's masked assignments without a renderer.
    env = object.__new__(cover_class)
    env.device = torch.device("cpu")
    env.num_envs = 3
    env.robot_uids = "panda_wristcam"
    env.robot_init_qpos_noise = 0.02
    env.reached_status = torch.zeros(3)
    env.scene = SimpleNamespace(_reset_mask=torch.ones(3, dtype=torch.bool))
    env.table_scene = SimpleNamespace(initialize=lambda indices: None)
    robot_qpos = torch.zeros(3, 9)

    class Actor:
        def __init__(self):
            self.pose = torch.zeros(3, 7)
            self.velocity = torch.zeros(3, 3)

        def set_pose(self, pose):
            self.pose[env.scene._reset_mask] = pose.raw_pose

        def set_linear_velocity(self, velocity):
            self.velocity[env.scene._reset_mask] = velocity

    env.ball, env.goal_region = Actor(), Actor()

    def reset_robot(qpos):
        robot_qpos[env.scene._reset_mask] = torch.as_tensor(qpos, dtype=torch.float32)

    env.agent = SimpleNamespace(
        reset=reset_robot, robot=SimpleNamespace(set_root_pose=lambda pose: None)
    )
    launches = []
    for seed in (3, 57, 101):
        env._episode_rng = np.random.RandomState(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            env._initialize_episode(torch.arange(3), {})
        expected = torch.tensor(cover_class.INITIAL_QPOS).expand(3, -1)
        torch.testing.assert_close(robot_qpos, expected, rtol=0, atol=0)
        launches.append(env.ball.pose.clone())
        assert torch.all((env.ball.pose[:, 0] >= -0.35) & (env.ball.pose[:, 0] <= -0.05))
        assert torch.all((env.ball.pose[:, 1] >= -0.90) & (env.ball.pose[:, 1] <= -0.65))
        assert torch.all((env.ball.velocity[:, 1] >= 0.75) & (env.ball.velocity[:, 1] <= 1))
    assert not torch.equal(launches[0], launches[1])
    assert not torch.equal(launches[1], launches[2])

    # A reset of row 1 must not move the arms or balls in the other episodes.
    robot_qpos[[0, 2]] += 0.1
    before_qpos, before_ball = robot_qpos.clone(), env.ball.pose.clone()
    env.scene._reset_mask[:] = torch.tensor([False, True, False])
    env._initialize_episode(torch.tensor([1]), {})
    torch.testing.assert_close(robot_qpos[[0, 2]], before_qpos[[0, 2]])
    torch.testing.assert_close(env.ball.pose[[0, 2]], before_ball[[0, 2]])
    torch.testing.assert_close(robot_qpos[1], torch.tensor(cover_class.INITIAL_QPOS))


def wrist_camera_transform(qpos):
    """Forward kinematics from the installed robot's actual URDF, without Vulkan."""
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
    from scipy.spatial.transform import Rotation

    root = ET.parse(PandaWristCam.urdf_path).getroot()
    poses = {"panda_link0": np.eye(4)}
    poses["panda_link0"][:3, 3] = [-0.615, 0, 0]
    for joint in root.findall("joint"):
        transform, motion = np.eye(4), np.eye(4)
        origin = joint.find("origin")
        if origin is not None:
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        if joint.get("type") == "revolute":
            angle = qpos[int(joint.get("name")[-1]) - 1]
            limits = joint.find("limit")
            assert float(limits.get("lower")) <= angle <= float(limits.get("upper"))
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        poses[child] = poses[parent] @ transform @ motion
    # PandaWristCam mounts its sensor at camera_link with an identity local pose.
    return poses["camera_link"]


def cover_bounds(cover_class):
    center, size = np.array(cover_class.COVER_CENTER), np.array(cover_class.COVER_SIZE)
    if cover_class.__name__ == "InterceptFastCover":
        return [(center - size / 2, center + size / 2)]
    section_size = size * [1, 1 / 3, 1]
    return [
        (center + offset - section_size / 2, center + offset + section_size / 2)
        for offset in (np.array([0, -size[1] / 3, 0]), np.array([0, size[1] / 3, 0]))
    ]


def intersects_box(eye, point, lower, upper):
    """Intersect the finite camera-to-point segment with an opaque cover."""
    entry, exit_ = 0.0, 1.0
    for axis, delta in enumerate(point - eye):
        if abs(delta) < 1e-12:
            if not lower[axis] <= eye[axis] <= upper[axis]:
                return False
            continue
        near, far = sorted(((lower[axis] - eye[axis]) / delta, (upper[axis] - eye[axis]) / delta))
        entry, exit_ = max(entry, near), min(exit_, far)
        if entry > exit_:
            return False
    return True


def assert_ball_visible(camera, center, cover_class):
    eye, rotation = camera[:3, 3], camera[:3, :3]
    local = rotation.T @ (center - eye)
    radius = cover_class.BALL_RADIUS
    # Keep the entire sphere inside the 90-degree frustum, with margin.
    assert local[0] > radius
    assert np.max(np.abs(local[1:])) + np.sqrt(2) * radius < local[0]
    # Check the center and surface landmarks, including the side nearest a
    # cover edge. Fitting the frustum alone does not establish visibility.
    for offset in (np.zeros(3), *np.eye(3) * radius, *np.eye(3) * -radius):
        assert not any(
            intersects_box(eye, center + offset, lower, upper)
            for lower, upper in cover_bounds(cover_class)
        ), (center, offset)


def test_launch_range_fits_camera_and_clears_cover_for_first_three_frames(cover_class):
    camera = wrist_camera_transform(cover_class.INITIAL_QPOS)
    eye, rotation = camera[:3, 3], camera[:3, :3]
    target = rotation.T @ (np.asarray(cover_class.INITIAL_LOOK_TARGET) - eye)
    # SAPIEN camera convention: +X forward, +Y left, +Z up.
    angle = np.arctan2(np.linalg.norm(target[1:]), target[0])
    assert np.degrees(angle) < 1

    # Include extreme initial positions and velocities at t=0, 0.05, 0.10 s.
    # z bounds cover the launch height and the ball settling onto the table.
    # This tests a held starting pose; later policy actions can move the camera.
    samples = itertools.product(
        (-0.35, -0.05), (-0.90, -0.65), (0, 0.05), (0.75, 1.0), (0.02, 0.04), (0, 0.05, 0.10)
    )
    for x, y, vx, vy, z, t in samples:
        center = np.array([x + vx * t, y + vy * t, z])
        assert_ball_visible(camera, center, cover_class)


def test_cover2_gap_and_launch_share_a_view_while_blocks_still_occlude():
    pytest.importorskip("mikasa_robo_suite")
    from dom_vpwem.custom_envs.intercept_fast_cover import InterceptFastCover
    from dom_vpwem.custom_envs.intercept_fast_cover2 import InterceptFastCover2

    camera = wrist_camera_transform(InterceptFastCover2.INITIAL_QPOS)
    np.testing.assert_allclose(camera[:3, 3], [-0.45, -0.45, 0.85], atol=1e-4)
    assert InterceptFastCover2.INITIAL_QPOS != InterceptFastCover.INITIAL_QPOS
    # Same unchanged camera transform for launch and the visible gap strip.
    for x in (-0.35, -0.20, -0.05):
        assert_ball_visible(camera, np.array([x, -0.775, 0.04]), InterceptFastCover2)
    for x, y in itertools.product((-0.35, -0.20, 0.0), (-0.30, -0.28, -0.26)):
        assert_ball_visible(camera, np.array([x, y, 0.02]), InterceptFastCover2)
    # The new vantage must retain the two occlusions, not see under the boxes.
    for y in (-13 / 30, -1 / 6):
        assert any(
            intersects_box(camera[:3, 3], np.array([-0.20, y, 0.02]), lower, upper)
            for lower, upper in cover_bounds(InterceptFastCover2)
        )


def test_wrist_camera_sees_ball_in_initial_frames_on_gpu(cover_class):
    if not torch.cuda.is_available():
        pytest.skip("A working NVIDIA GPU and Vulkan renderer are required")
    import gymnasium as gym

    from dom_vpwem.custom_envs import register_custom_envs

    register_custom_envs()
    env = gym.make(
        f"{cover_class.__name__}-VLA-v0",
        num_envs=4,
        obs_mode="rgb+segmentation",
        control_mode="pd_ee_delta_pose",
        sim_backend="gpu",
        reconfiguration_freq=0,
    )
    try:
        base = env.unwrapped
        initial_pose = None
        for seeds in ([0, 1, 2, 3], [41, 42, 43, 44]):
            obs, _ = env.reset(seed=seeds)
            qpos = torch.tensor(cover_class.INITIAL_QPOS, device=base.device).expand(4, -1)
            torch.testing.assert_close(base.agent.robot.get_qpos(), qpos)
            camera_pose = base.agent.robot.links_map["camera_link"].pose.raw_pose.clone()
            if initial_pose is not None:
                torch.testing.assert_close(camera_pose, initial_pose)
            initial_pose = camera_pose
            for frame in range(3):
                segmentation = obs["sensor_data"]["hand_camera"]["segmentation"]
                ball_pixels = segmentation == base.ball.per_scene_id[:, None, None, None]
                assert torch.all(ball_pixels.flatten(1).sum(1) >= 4), (seeds, frame)
                if frame < 2:
                    obs, _, _, _, _ = env.step(torch.zeros(4, 7, device=base.device))
            if cover_class.__name__ == "InterceptFastCover2":
                from mani_skill.utils.structs.pose import Pose

                # Move the test ball into the gap without changing the camera
                # or advancing physics, and verify actual rendered visibility.
                xyz = torch.zeros(4, 3, device=base.device)
                xyz[:, 0] = torch.linspace(-0.35, 0, 4, device=base.device)
                xyz[:, 1], xyz[:, 2] = -0.30, base.BALL_RADIUS
                base.ball.set_pose(Pose.create_from_pq(p=xyz))
                base.scene._gpu_apply_all()
                base.scene._gpu_fetch_all()
                gap_obs = base.get_obs()
                segmentation = gap_obs["sensor_data"]["hand_camera"]["segmentation"]
                ball_pixels = segmentation == base.ball.per_scene_id[:, None, None, None]
                assert torch.all(ball_pixels.flatten(1).sum(1) >= 4)
        # Verify real simulator subset resets also restore only the selected arm.
        before = base.agent.robot.get_qpos().clone()
        env.reset(options={"env_idx": torch.tensor([1], device=base.device)})
        torch.testing.assert_close(base.agent.robot.get_qpos()[[0, 2, 3]], before[[0, 2, 3]])
        torch.testing.assert_close(base.agent.robot.get_qpos()[1], qpos[1])
    finally:
        env.close()
