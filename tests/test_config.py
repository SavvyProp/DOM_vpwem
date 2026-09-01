from pathlib import Path

import pytest

from dom_vpwem.config import ExperimentConfig, ModelConfig, TaskConfig, TrainConfig


def test_task_config_loads_and_preserves_vpwem_alignment() -> None:
    config = ExperimentConfig.from_yaml(
        Path("configs/shell_game_shuffle_color_lamp_touch.yaml")
    )
    assert config.task.env_id == "ShellGameShuffleColorLampTouch-VLA-v0"
    assert config.task.max_episode_steps == 60
    assert config.model.action_dim == 7
    assert config.model.proprio_dim == 7
    assert config.model.action_horizon == 9
    assert config.model.action_horizon == (
        config.model.obs_steps + config.model.action_steps - 1
    )
    assert config.model.embedding_dropout == 0.0
    assert config.model.memory_cache_strategy == "fifo"


def test_model_config_rejects_wrong_action_contract() -> None:
    with pytest.raises(ValueError, match="7-D"):
        ModelConfig(action_dim=8).validate()


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown ModelConfig"):
        ExperimentConfig.from_dict({"model": {"not_a_setting": True}})


def test_practical_training_defaults() -> None:
    train = TrainConfig()

    assert train.batch_size == 16
    assert train.num_workers == 2
    assert train.prefetch_factor == 1
    assert train.gradient_steps == 600_000
    assert train.vision_encoder_checkpoint is None


def test_prefetch_factor_must_be_positive() -> None:
    config = ExperimentConfig()
    config.train.prefetch_factor = 0

    with pytest.raises(ValueError, match="prefetch_factor"):
        config.validate()


def test_frozen_random_encoder_is_rejected_without_a_checkpoint() -> None:
    config = ExperimentConfig()
    config.model.freeze_vision_encoder = True

    with pytest.raises(ValueError, match="vision_encoder_checkpoint"):
        config.validate()

    config.train.vision_encoder_checkpoint = "encoder.pt"
    config.validate()


def test_full_resume_is_valid_initialization_for_a_frozen_encoder() -> None:
    config = ExperimentConfig()
    config.model.freeze_vision_encoder = True
    config.train.resume = "full_vpwem.pt"

    config.validate()


def test_shell_game_touch_task_derives_registry_defaults() -> None:
    task = TaskConfig(env_id="ShellGameTouch-VLA-v0")

    assert task.dataset_dir == "data_mikasa_robo/data_npz/shell_game_touch_vla_v0"
    assert task.max_episode_steps == 30
    ExperimentConfig(task=task).validate()


def test_task_horizon_must_match_registered_benchmark() -> None:
    task = TaskConfig(
        env_id="ShellGameTouch-VLA-v0",
        max_episode_steps=60,
    )

    with pytest.raises(ValueError, match="30-step horizon"):
        ExperimentConfig(task=task).validate()
