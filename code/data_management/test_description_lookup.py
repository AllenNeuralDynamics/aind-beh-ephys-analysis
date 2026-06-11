"""
Test that description lookup works correctly with original column names
"""
import json

# Load files
with open('/root/capsule/code/data_management/column_names_map.json', 'r') as f:
    column_map = json.load(f)

with open('/root/capsule/code/data_management/column_names_description.json', 'r') as f:
    column_desc = json.load(f)

# Create reverse mapping
mapped_to_original = {}
for orig_col, mapped_name in column_map['unit_columns_custom'].items():
    if 'duplicate as' in mapped_name:
        continue
    if 'similar to' in mapped_name:
        clean_name = mapped_name.split(';')[0].strip()
        mapped_to_original[clean_name] = orig_col
    else:
        mapped_to_original[mapped_name] = orig_col

for orig_col, mapped_name in column_map['unit_columns_ks'].items():
    if mapped_name not in mapped_to_original:
        mapped_to_original[mapped_name] = orig_col

print("="*80)
print("TESTING DESCRIPTION LOOKUP")
print("="*80)

# Test some examples
test_columns = [
    'unit_id',
    'maximum_increase_of_p(response|laser)_from_baseline',
    'opto_tagged_lowbar',
    'firing_rate',
    'spike_times',
    'electrodes',
    'peak_trough_ratio',
]

unit_descriptions = column_desc.get('unit_columns', {})

print("\nTesting description lookup for mapped column names:")
for col in test_columns:
    orig_col = mapped_to_original.get(col, col)
    description = unit_descriptions.get(orig_col, col)
    if description == 'to be filled':
        description = f"[TO BE FILLED] {col}"

    print(f"\nMapped column: {col}")
    print(f"  -> Original: {orig_col}")
    print(f"  -> Description: {description}")

# Count coverage
print("\n" + "="*80)
print("COVERAGE SUMMARY")
print("="*80)

found = 0
to_be_filled = 0
missing = 0

for mapped_col, orig_col in mapped_to_original.items():
    if orig_col in unit_descriptions:
        desc = unit_descriptions[orig_col]
        if desc == 'to be filled':
            to_be_filled += 1
        else:
            found += 1
    else:
        missing += 1
        print(f"Missing: {mapped_col} <- {orig_col}")

total = len(mapped_to_original)
print(f"\nTotal mapped columns: {total}")
print(f"  ✓ Found with descriptions: {found}")
print(f"  ⚠ Found but 'to be filled': {to_be_filled}")
print(f"  ✗ Missing from description file: {missing}")
print(f"\nCoverage: {(found + to_be_filled) / total * 100:.1f}%")
