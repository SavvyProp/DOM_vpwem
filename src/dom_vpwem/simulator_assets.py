"""Install and verify simulator assets omitted from upstream package artifacts.

MIKASA-Robo 1.0.0 references a lamp mesh that exists in its tagged Git
source, but the file was not included in either its wheel or source
distribution.  This module retrieves that exact release asset without
hard-coding a virtual-environment or Python-version path.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import BinaryIO, Sequence

MIKASA_DISTRIBUTION_NAME = "mikasa-robo-suite"
MIKASA_RELEASE_COMMIT = "16634db18bef08128ed79346469c86fc12169aed"
MIKASA_LAMP_ASSET_RELATIVE_PATH = Path(
    "mikasa_robo_suite/vla/utils/objects/low_poly_light_bulb.glb"
)
MIKASA_LAMP_ASSET_URL = (
    "https://raw.githubusercontent.com/CognitiveAISystems/MIKASA-Robo/"
    f"{MIKASA_RELEASE_COMMIT}/{MIKASA_LAMP_ASSET_RELATIVE_PATH.as_posix()}"
)
MIKASA_LAMP_ASSET_SHA256 = "cc66bda617401ce47bf4bd69e297c226b909d2bb532a5ff57721ee75d6d4aca5"
MIKASA_LAMP_ASSET_SIZE = 697_948
INSTALL_COMMAND = "uv run --locked --extra eval dom-vpwem-install-sim-assets"


class SimulatorAssetError(RuntimeError):
    """Raised when a required simulator asset cannot be installed or verified."""


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def mikasa_lamp_asset_path() -> Path:
    """Return the lamp path inside the installed MIKASA distribution."""

    try:
        distribution = metadata.distribution(MIKASA_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError as exc:
        raise SimulatorAssetError(
            "mikasa-robo-suite is not installed. Run `uv sync --locked --extra eval` first."
        ) from exc
    return Path(distribution.locate_file(str(MIKASA_LAMP_ASSET_RELATIVE_PATH)))


def _asset_problem(target: Path, detail: str) -> SimulatorAssetError:
    return SimulatorAssetError(
        f"MIKASA lamp asset {detail}: {target}. MIKASA-Robo 1.0.0 omitted "
        "this file from its package artifacts. Install the verified asset with "
        f"`{INSTALL_COMMAND}`."
    )


def require_mikasa_lamp_asset() -> Path:
    """Return the installed lamp asset or raise with an actionable remedy."""

    target = mikasa_lamp_asset_path()
    if not target.is_file():
        raise _asset_problem(target, "is missing")
    digest, size = _sha256(target)
    if digest != MIKASA_LAMP_ASSET_SHA256:
        raise _asset_problem(
            target,
            "failed SHA-256 verification "
            f"(expected {MIKASA_LAMP_ASSET_SHA256}, found {digest}; {size} bytes)",
        )
    return target


def _copy_stream(stream: BinaryIO, destination: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as output:
        while chunk := stream.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest(), size


def install_mikasa_sim_assets(
    *,
    source: Path | None = None,
    force: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Install the checksum-pinned lamp mesh into the MIKASA package.

    ``source`` may name a previously downloaded copy for an offline install.
    Existing correct content is an idempotent no-op. Unexpected content is
    preserved unless ``force`` is explicitly requested.
    """

    if timeout <= 0:
        raise SimulatorAssetError(f"timeout must be positive, got {timeout}.")

    target = mikasa_lamp_asset_path()
    if target.exists():
        if not target.is_file():
            raise SimulatorAssetError(
                f"Cannot install MIKASA lamp asset: target is not a file: {target}"
            )
        digest, size = _sha256(target)
        if digest == MIKASA_LAMP_ASSET_SHA256:
            return target
        if not force:
            raise SimulatorAssetError(
                f"Refusing to replace existing MIKASA lamp asset at {target}: "
                f"expected SHA-256 {MIKASA_LAMP_ASSET_SHA256}, found {digest} "
                f"({size} bytes). Re-run with --force only if replacing this "
                "file is intentional."
            )

    local_source: Path | None = None
    if source is not None:
        try:
            local_source = source.expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise SimulatorAssetError(f"Local asset source does not exist: {source}") from exc
        if not local_source.is_file():
            raise SimulatorAssetError(f"Local asset source is not a file: {local_source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".partial",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if local_source is not None:
            with local_source.open("rb") as stream:
                digest, size = _copy_stream(stream, temporary)
        else:
            request = urllib.request.Request(
                MIKASA_LAMP_ASSET_URL,
                headers={"User-Agent": "dom-vpwem-simulator-asset-installer/0.1"},
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as stream:
                    digest, size = _copy_stream(stream, temporary)
            except (OSError, urllib.error.URLError) as exc:
                raise SimulatorAssetError(
                    f"Failed to download MIKASA lamp asset from {MIKASA_LAMP_ASSET_URL}: {exc}"
                ) from exc

        if digest != MIKASA_LAMP_ASSET_SHA256:
            raise SimulatorAssetError(
                "MIKASA lamp asset failed SHA-256 verification: "
                f"expected {MIKASA_LAMP_ASSET_SHA256}, found {digest} "
                f"({size} bytes). The destination was not changed."
            )
        if size != MIKASA_LAMP_ASSET_SIZE:
            # A matching SHA-256 already proves content identity. Keep the
            # explicit size assertion as release metadata and defense in depth.
            raise SimulatorAssetError(
                "MIKASA lamp asset has an unexpected size despite matching its "
                f"digest: expected {MIKASA_LAMP_ASSET_SIZE}, found {size}."
            )

        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    return target


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install the checksum-pinned lamp mesh omitted from "
            "mikasa-robo-suite 1.0.0 package artifacts."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Use a local GLB instead of downloading it (the same SHA-256 is required).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file only after the replacement passes verification.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Network timeout in seconds (default: 60).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        target = install_mikasa_sim_assets(
            source=args.source,
            force=args.force,
            timeout=args.timeout,
        )
    except SimulatorAssetError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Verified MIKASA lamp asset: {target}")
    print(f"SHA-256: {MIKASA_LAMP_ASSET_SHA256}")
    print('Attribution: "Low Poly Light Bulb" by AleixoAlonso, licensed CC BY 4.0.')
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "INSTALL_COMMAND",
    "MIKASA_LAMP_ASSET_RELATIVE_PATH",
    "MIKASA_LAMP_ASSET_SHA256",
    "MIKASA_LAMP_ASSET_SIZE",
    "MIKASA_LAMP_ASSET_URL",
    "SimulatorAssetError",
    "install_mikasa_sim_assets",
    "mikasa_lamp_asset_path",
    "require_mikasa_lamp_asset",
]
