"""
Test script to merge unit tables from custom pickle and NWB kilosort data.
"""
import json
import pandas as pd
import numpy as np
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
import sys
import os

# Add beh_ephys_analysis to path
sys.path.insert(0, '/root/capsule/code/beh_ephys_analysis')
from utils.beh_functions import get_unit_tbl, session_dirs

# Load column mappings
with open('/root/capsule/code/data_management/column_names_map.json', 'r') as f:
    column_map = json.load(f)

unit_columns_custom = column_map['unit_columns_custom']
unit_columns_ks = column_map['unit_columns_ks']

def merge_unit_tables(session_id, data_type='curated'):
    """
    Merge unit tables from custom pickle and NWB kilosort data.

    Returns merged DataFrame with mapped column names.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {session_id}")
    print(f"{'='*80}\n")

    # 1. Load custom unit table from pickle (use summary version)
    print("1. Loading custom unit table from get_unit_tbl (summary=True)...")
    custom_unit_tbl = get_unit_tbl(session_id, data_type=data_type, summary=True)
    if custom_unit_tbl is None:
        print("  ERROR: No custom unit table found")
        return None
    print(f"  ✓ Loaded {len(custom_unit_tbl)} units")
    print(f"  Columns: {list(custom_unit_tbl.columns)[:10]}...")

    # 2. Load NWB kilosort data
    print("\n2. Loading NWB kilosort data...")
    session_dir = session_dirs(session_id)
    nwb_path = session_dir.get(f'nwb_dir_{data_type}')
    if nwb_path is None or not os.path.exists(nwb_path):
        print(f"  ERROR: NWB file not found at {nwb_path}")
        return None

    nwb = load_nwb_from_filename(nwb_path)
    nwb_unit_tbl = nwb.units.to_dataframe()
    print(f"  ✓ Loaded {len(nwb_unit_tbl)} units from NWB")
    print(f"  Columns: {list(nwb_unit_tbl.columns)[:10]}...")

    # 3. Verify row counts match
    print("\n3. Verifying row counts...")
    custom_unit_ids = set(custom_unit_tbl['unit_id'].values)
    nwb_ks_unit_ids = set(nwb_unit_tbl['ks_unit_id'].values)

    common_ids = custom_unit_ids & nwb_ks_unit_ids
    only_custom = custom_unit_ids - nwb_ks_unit_ids
    only_nwb = nwb_ks_unit_ids - custom_unit_ids

    print(f"  Custom table: {len(custom_unit_tbl)} units")
    print(f"  NWB table: {len(nwb_unit_tbl)} units")
    print(f"  Common unit_ids: {len(common_ids)}")

    if only_custom:
        print(f"  ⚠ WARNING: {len(only_custom)} units only in custom: {sorted(list(only_custom))[:10]}...")
    if only_nwb:
        print(f"  ⚠ WARNING: {len(only_nwb)} units only in NWB: {sorted(list(only_nwb))[:10]}...")

    if len(custom_unit_tbl) != len(common_ids):
        print(f"  ⚠ WARNING: Not all custom units found in NWB!")

    # 4. Align tables by unit_id
    print("\n4. Aligning tables...")
    # Filter to common units and sort by unit_id
    custom_aligned = custom_unit_tbl[custom_unit_tbl['unit_id'].isin(common_ids)].sort_values('unit_id').reset_index(drop=True)
    nwb_aligned = nwb_unit_tbl[nwb_unit_tbl['ks_unit_id'].isin(common_ids)].sort_values('ks_unit_id').reset_index(drop=True)

    print(f"  ✓ Aligned {len(custom_aligned)} common units")

    # 5. Apply column mappings and merge
    print("\n5. Applying column mappings...")
    merged_df = pd.DataFrame()

    # Add custom columns with mapped names
    for orig_col, mapped_name in unit_columns_custom.items():
        if orig_col not in custom_aligned.columns:
            continue

        # Skip duplicates (they'll be taken from NWB)
        if 'duplicate as' in mapped_name:
            print(f"  - Skipping '{orig_col}' (duplicate)")
            continue

        # Handle similar columns
        if 'similar to' in mapped_name:
            # Use the descriptive name before "similar"
            clean_name = mapped_name.split(';')[0].strip()
            merged_df[clean_name] = custom_aligned[orig_col].values
            print(f"  + Added '{orig_col}' as '{clean_name}' (similar)")
        else:
            merged_df[mapped_name] = custom_aligned[orig_col].values
            print(f"  + Added '{orig_col}' as '{mapped_name}'")

    # Add NWB columns with mapped names
    for orig_col, mapped_name in unit_columns_ks.items():
        if orig_col not in nwb_aligned.columns:
            continue

        # Check if this column name already exists in merged_df
        if mapped_name in merged_df.columns:
            print(f"  - Skipping '{orig_col}' (already exists as '{mapped_name}')")
            continue

        merged_df[mapped_name] = nwb_aligned[orig_col].values
        print(f"  + Added '{orig_col}' as '{mapped_name}'")

    print(f"\n✓ Merged table has {len(merged_df)} rows and {len(merged_df.columns)} columns")

    return merged_df

# Test with example sessions
if __name__ == '__main__':
    sessions = [
        'behavior_754897_2025-03-13_11-20-42',
        'behavior_ZS061_2021-04-08_18-01-30',
    ]

    for session in sessions:
        try:
            merged = merge_unit_tables(session, data_type='curated')
            if merged is not None:
                print(f"\nFinal merged columns ({len(merged.columns)}):")
                for i, col in enumerate(merged.columns, 1):
                    print(f"  {i:3d}. {col}")
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
