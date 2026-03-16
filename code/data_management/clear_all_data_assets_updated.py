# %%
from codeocean import CodeOcean
import os

client = CodeOcean(
    domain="https://codeocean.allenneuraldynamics.org",
    token=os.getenv("API_SECRET"),
)

# %%
computation_id = os.getenv("CO_COMPUTATION_ID")
if not computation_id:
    raise ValueError("CO_COMPUTATION_ID is not set in the environment.")

# Read the current computation so we can see which data assets are attached
computation = client.computations.get_computation(computation_id=computation_id)

ids = []
if computation and getattr(computation, "data_assets", None):
    ids = list({data_asset.id for data_asset in computation.data_assets})

print(f"Found {len(ids)} attached data assets.")

# %%
if ids:
    results_rm = client.computations.detach_data_assets(
        computation_id=computation_id,
        data_assets=ids,
    )
    print(f"Removed all {len(ids)} data assets from computation {computation_id}.")
else:
    print("No data assets to remove.")

# %%
# Check what remains attached
computation = client.computations.get_computation(computation_id=computation_id)

remaining_ids = []
if computation and getattr(computation, "data_assets", None):
    remaining_ids = list({data_asset.id for data_asset in computation.data_assets})

print(f"Now {len(remaining_ids)} data assets remain.")
# %%