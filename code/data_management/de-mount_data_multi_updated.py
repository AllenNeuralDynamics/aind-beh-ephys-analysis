# %%
import pandas as pd
import os
from codeocean import CodeOcean

client = CodeOcean(
    domain="https://codeocean.allenneuraldynamics.org",
    token=os.getenv("API_SECRET"),
)

# %%
# load and parse data ids
script_dir = os.path.dirname(os.path.abspath(__file__))
datalist_dir = os.path.join(script_dir, "session_assets.csv")
data_df = pd.read_csv(datalist_dir)

# %%
col_to_remove = ["raw_data", "sorted_curated", "model_stan"]

# %%
# Collect data asset IDs to detach
all_ids = []

for curr_col in col_to_remove:
    if curr_col not in data_df.columns:
        print(f"Warning: column '{curr_col}' not found, skipping.")
        continue

    curr_ids = [
        s for s in data_df[curr_col].tolist()
        if isinstance(s, str) and 30 < len(s) < 40
    ]
    all_ids.extend(curr_ids)

# Optional: remove duplicates while preserving order
data_assets_id = list(dict.fromkeys(all_ids))

# %%
# Detach from the currently running computation / cloud workstation
computation_id = os.getenv("CO_COMPUTATION_ID")
if not computation_id:
    raise ValueError("CO_COMPUTATION_ID is not set in the environment.")

results_rm = client.computations.detach_data_assets(
    computation_id=computation_id,
    data_assets=data_assets_id,
)

print(f"Removed {len(data_assets_id)} data assets from computation {computation_id}.")