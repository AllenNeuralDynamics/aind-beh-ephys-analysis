"""Attach all required data assets to a Code Ocean capsule for reproducible runs.

This script mirrors the asset-selection logic in:
- mount_data_multi.py
- mount_data_multi_hopkins.py

Unlike those scripts, this script always attaches to a capsule via CO_CAPSULE_ID.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from codeocean import CodeOcean
from codeocean.data_asset import DataAssetAttachParams

os.sys.path.append(str(Path(__file__).resolve().parent.parent / "beh_ephys_analysis"))
from utils.beh_functions import parseSessionID


def _is_asset_id(value: object) -> bool:
    return isinstance(value, str) and 30 < len(value) < 40


def _normalize_mount_name(mount: str) -> str:
    """Match the legacy `mount_data_multi*.py` handling of stan-model mounts."""
    if "stan" not in mount:
        return mount

    session_prefix = mount.split("_model_stan")[0]
    ani_id, _, _ = parseSessionID(session_prefix)
    if ani_id is None:
        return mount
    return f"{ani_id}_model_stan"


def _load_mount_data_multi_assets(base_dir: Path) -> list[tuple[str, str]]:
    """Collect assets from session_assets.csv with mount names used in mount_data_multi.py."""
    data_df = pd.read_csv(base_dir / "session_assets.csv")
    data_df = data_df[data_df["session_id"].notna() & (data_df["session_id"] != "")]

    col_to_attach = ["raw_data", "sorted_curated", "sorted", "model_stan"]
    pairs: list[tuple[str, str]] = []

    for curr_col in col_to_attach:
        valid_mask = [_is_asset_id(v) for v in data_df[curr_col].to_list()]
        session_ids = list(data_df[valid_mask]["session_id"].values)
        curr_ids = list(data_df[valid_mask][curr_col].values)

        for session_id, asset_id in zip(session_ids, curr_ids):
            mount = f"{session_id}_{curr_col}"
            mount = _normalize_mount_name(mount)
            pairs.append((asset_id, mount))

    # Extra static assets loaded from combined_assets.csv.
    combined_df = pd.read_csv(base_dir / "combined_assets.csv")
    combined_df.columns = combined_df.columns.str.strip()
    combined_df["id"] = combined_df["id"].str.strip().str.strip('"')
    combined_df["mount_name"] = combined_df["mount_name"].str.strip().str.strip('"')
    pairs.extend(list(zip(combined_df["id"].tolist(), combined_df["mount_name"].tolist())))

    return pairs


def _load_mount_data_multi_hopkins_assets(base_dir: Path) -> list[tuple[str, str]]:
    """Collect assets with mount names used in mount_data_multi_hopkins.py."""
    datalist_files = [
        base_dir / "hopkins_session_assets.csv",
        base_dir / "hopkins_FP_session_assets.csv",
    ]
    dfs = [pd.read_csv(path) for path in datalist_files]
    data_df = pd.concat(dfs, ignore_index=True)
    data_df = data_df[data_df["session_id"].notna() & (data_df["session_id"] != "")]

    pairs: list[tuple[str, str]] = []

    valid_mask = [_is_asset_id(v) for v in data_df["raw_data"].to_list()]
    session_ids = list(data_df[valid_mask]["session_id"].values)
    curr_ids = list(data_df[valid_mask]["raw_data"].values)
    pairs.extend((asset_id, f"{session_id}_raw_data") for session_id, asset_id in zip(session_ids, curr_ids))

    model_df = pd.read_csv(base_dir / "hopkins_model_assets.csv")
    valid_mask = [_is_asset_id(v) for v in model_df["model_stan"].to_list()]
    animal_ids = list(model_df[valid_mask]["animal_id"].values)
    curr_ids = list(model_df[valid_mask]["model_stan"].values)
    pairs.extend((asset_id, f"{animal_id}_model_stan") for animal_id, asset_id in zip(animal_ids, curr_ids))

    return pairs


def _dedupe_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove exact duplicates while preserving order."""
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    capsule_id = os.getenv("CO_CAPSULE_ID")
    if not capsule_id:
        raise RuntimeError("CO_CAPSULE_ID is not set. This script is for capsule reproducible runs.")

    token = os.getenv("API_SECRET")
    if not token:
        raise RuntimeError("API_SECRET is not set.")

    client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=token)

    pairs = _load_mount_data_multi_hopkins_assets(script_dir) + _load_mount_data_multi_assets(script_dir)
    pairs = _dedupe_pairs(pairs)

    attach_params = [DataAssetAttachParams(id=asset_id, mount=mount) for asset_id, mount in pairs]

    print(f"Attaching {len(attach_params)} assets to capsule {capsule_id}...")
    results = client.capsules.attach_data_assets(
        capsule_id=capsule_id,
        attach_params=attach_params,
    )

    print(f"Attached {len(results)} data assets.")
    for data_asset in results:
        print(f"{data_asset.id} mounted as {data_asset.mount}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
