"""
Check if all columns in column_names_map.json are covered in column_names_description.json
"""
import json

# Load files
with open('/root/capsule/code/data_management/column_names_map.json', 'r') as f:
    column_map = json.load(f)

with open('/root/capsule/code/data_management/column_names_description.json', 'r') as f:
    column_desc = json.load(f)

print("="*80)
print("COLUMN COVERAGE CHECK")
print("="*80)

# Check behavior_trial_columns
print("\n1. Checking behavior_trial_columns...")
trial_mapped = set(column_map['behavior_trial_columns'].values())
trial_described = set(column_desc['behavior_trial_columns'].keys())

missing_trial = trial_mapped - trial_described
extra_trial = trial_described - trial_mapped

if missing_trial:
    print(f"  ✗ Missing descriptions for {len(missing_trial)} columns:")
    for col in sorted(missing_trial):
        print(f"    - {col}")
else:
    print(f"  ✓ All {len(trial_mapped)} trial columns have descriptions")

if extra_trial:
    print(f"  ⚠ {len(extra_trial)} extra columns in descriptions (not in map):")
    for col in sorted(extra_trial):
        print(f"    - {col}")

# Check unit_columns_custom
print("\n2. Checking unit_columns_custom...")
unit_custom_mapped = set()
for orig_col, mapped_name in column_map['unit_columns_custom'].items():
    # Skip duplicates
    if 'duplicate as' in mapped_name:
        continue
    # Handle similar columns
    if 'similar to' in mapped_name:
        clean_name = mapped_name.split(';')[0].strip()
        unit_custom_mapped.add(clean_name)
    else:
        unit_custom_mapped.add(mapped_name)

print(f"  Custom columns after mapping: {len(unit_custom_mapped)}")

# Check unit_columns_ks
print("\n3. Checking unit_columns_ks...")
unit_ks_mapped = set(column_map['unit_columns_ks'].values())
print(f"  KS columns: {len(unit_ks_mapped)}")

# Combined unit columns
all_unit_mapped = unit_custom_mapped | unit_ks_mapped
unit_described = set(column_desc['unit_columns'].keys())

missing_unit = all_unit_mapped - unit_described
extra_unit = unit_described - all_unit_mapped

print(f"\n4. Combined unit column coverage...")
print(f"  Total mapped unit columns: {len(all_unit_mapped)}")
print(f"  Total described unit columns: {len(unit_described)}")

if missing_unit:
    print(f"  ✗ Missing descriptions for {len(missing_unit)} columns:")
    for col in sorted(missing_unit):
        print(f"    - {col}")
else:
    print(f"  ✓ All {len(all_unit_mapped)} unit columns have entries in descriptions")

if extra_unit:
    print(f"  ⚠ {len(extra_unit)} extra columns in descriptions (not in map):")
    for col in sorted(extra_unit):
        print(f"    - {col}")

# Check for "to be filled"
print(f"\n5. Checking for unfilled descriptions...")
unfilled_unit = [col for col, desc in column_desc['unit_columns'].items()
                 if desc == 'to be filled' and col in all_unit_mapped]

if unfilled_unit:
    print(f"  ⚠ {len(unfilled_unit)} unit columns need descriptions (currently 'to be filled'):")
    for col in sorted(unfilled_unit):
        print(f"    - {col}")
else:
    print(f"  ✓ All unit columns have actual descriptions")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Trial columns: {'✓ COMPLETE' if not missing_trial else f'✗ {len(missing_trial)} MISSING'}")
print(f"Unit columns: {'✓ ALL HAVE ENTRIES' if not missing_unit else f'✗ {len(missing_unit)} MISSING'}")
print(f"Unfilled unit descriptions: {len(unfilled_unit)}")
print("="*80)
