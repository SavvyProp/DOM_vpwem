from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from dom_vpwem import simulator_assets
from dom_vpwem.simulator_assets import (
    MIKASA_LAMP_ASSET_RELATIVE_PATH,
    SimulatorAssetError,
    install_mikasa_sim_assets,
    mikasa_lamp_asset_path,
    require_mikasa_lamp_asset,
)


def _set_expected_payload(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(
        simulator_assets,
        "MIKASA_LAMP_ASSET_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(simulator_assets, "MIKASA_LAMP_ASSET_SIZE", len(payload))


def test_distribution_location_is_resolved_dynamically(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    class FakeDistribution:
        def locate_file(self, relative_path: str) -> Path:
            assert relative_path == str(MIKASA_LAMP_ASSET_RELATIVE_PATH)
            return tmp_path / relative_path

    def distribution(name: str) -> FakeDistribution:
        calls.append(name)
        return FakeDistribution()

    monkeypatch.setattr(simulator_assets.metadata, "distribution", distribution)

    assert mikasa_lamp_asset_path() == tmp_path / MIKASA_LAMP_ASSET_RELATIVE_PATH
    assert calls == ["mikasa-robo-suite"]


def test_remote_install_is_verified_atomic_and_idempotent(monkeypatch, tmp_path) -> None:
    payload = b"test lamp asset"
    target = tmp_path / "site-packages/mikasa_robo_suite/vla/utils/objects/lamp.glb"
    _set_expected_payload(monkeypatch, payload)
    monkeypatch.setattr(simulator_assets, "mikasa_lamp_asset_path", lambda: target)
    requests = []

    def urlopen(request, *, timeout):
        requests.append((request.full_url, request.headers, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(simulator_assets.urllib.request, "urlopen", urlopen)

    assert install_mikasa_sim_assets(timeout=12.5) == target
    assert target.read_bytes() == payload
    assert requests[0][0] == simulator_assets.MIKASA_LAMP_ASSET_URL
    assert requests[0][2] == 12.5
    assert not list(target.parent.glob("*.partial"))

    monkeypatch.setattr(
        simulator_assets.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("idempotent install accessed the network"),
    )
    assert install_mikasa_sim_assets() == target
    assert require_mikasa_lamp_asset() == target


def test_local_source_is_supported_without_network(monkeypatch, tmp_path) -> None:
    payload = b"offline lamp asset"
    source = tmp_path / "downloaded.glb"
    source.write_bytes(payload)
    target = tmp_path / "site-packages/mikasa_robo_suite/vla/utils/objects/lamp.glb"
    _set_expected_payload(monkeypatch, payload)
    monkeypatch.setattr(simulator_assets, "mikasa_lamp_asset_path", lambda: target)
    monkeypatch.setattr(
        simulator_assets.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("local install accessed the network"),
    )

    assert install_mikasa_sim_assets(source=source) == target
    assert target.read_bytes() == payload


def test_existing_unexpected_file_is_preserved_unless_forced(monkeypatch, tmp_path) -> None:
    replacement = b"verified replacement"
    existing = b"user supplied content"
    source = tmp_path / "replacement.glb"
    source.write_bytes(replacement)
    target = tmp_path / "objects/lamp.glb"
    target.parent.mkdir(parents=True)
    target.write_bytes(existing)
    _set_expected_payload(monkeypatch, replacement)
    monkeypatch.setattr(simulator_assets, "mikasa_lamp_asset_path", lambda: target)

    with pytest.raises(SimulatorAssetError, match="Refusing to replace"):
        install_mikasa_sim_assets(source=source)
    assert target.read_bytes() == existing

    assert install_mikasa_sim_assets(source=source, force=True) == target
    assert target.read_bytes() == replacement


def test_bad_source_never_replaces_existing_file(monkeypatch, tmp_path) -> None:
    expected = b"expected asset"
    existing = b"keep me"
    source = tmp_path / "bad.glb"
    source.write_bytes(b"not the expected asset")
    target = tmp_path / "objects/lamp.glb"
    target.parent.mkdir(parents=True)
    target.write_bytes(existing)
    _set_expected_payload(monkeypatch, expected)
    monkeypatch.setattr(simulator_assets, "mikasa_lamp_asset_path", lambda: target)

    with pytest.raises(SimulatorAssetError, match="failed SHA-256"):
        install_mikasa_sim_assets(source=source, force=True)

    assert target.read_bytes() == existing
    assert not list(target.parent.glob("*.partial"))


def test_preflight_reports_actionable_install_command(monkeypatch, tmp_path) -> None:
    target = tmp_path / "missing.glb"
    monkeypatch.setattr(simulator_assets, "mikasa_lamp_asset_path", lambda: target)

    with pytest.raises(
        SimulatorAssetError,
        match="dom-vpwem-install-sim-assets",
    ):
        require_mikasa_lamp_asset()
