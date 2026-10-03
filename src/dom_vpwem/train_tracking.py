"""Train the separate tracking-supervised VPWEM on custom shuffle touch."""

from pathlib import Path

from .tracking import TrackingExperimentConfig, TrackingVPWEM
from .tracking_data import TrackingNpzDataset
from .train import build_arg_parser, prepare_batch, train


def prepare_tracking_batch(batch, stats, device):
    prepared = prepare_batch(batch, stats, device)
    for key in ("tracking_xy", "tracking_hidden", "tracking_mask"):
        prepared[key] = batch[key].to(device=device, non_blocking=True)
    return prepared


def main(argv=None):
    parser = build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(config=Path("configs/shell_game_shuffle_touch_tracking.yaml"))
    for action in parser._actions:
        if action.dest == "config":
            action.help = "Tracking YAML (default: configs/shell_game_shuffle_touch_tracking.yaml)."
    args = parser.parse_args(argv)
    config = TrackingExperimentConfig.from_yaml(args.config)
    for arg, field in (
        ("output_dir", "output_dir"), ("device", "device"), ("steps", "gradient_steps"),
        ("batch_size", "batch_size"), ("resume", "resume"),
        ("vision_encoder_checkpoint", "vision_encoder_checkpoint"),
    ):
        value = getattr(args, arg)
        if value is not None:
            setattr(config.train, field, str(value) if isinstance(value, Path) else value)
    if args.dataset_dir is not None:
        config.task.dataset_dir = str(args.dataset_dir)
    config.validate()
    checkpoint = train(
        config, dataset_factory=TrackingNpzDataset,
        model_factory=lambda model: TrackingVPWEM(model, config.tracking),
        batch_preparer=prepare_tracking_batch,
    )
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
