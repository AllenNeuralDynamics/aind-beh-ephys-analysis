# %%
from codeocean import CodeOcean
import os
client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=os.getenv("API_SECRET"))

# %%
computations = client.capsules.list_computations(capsule_id=os.getenv("CO_CAPSULE_ID"))
ids = list(set([data_asset.id for computation in computations for data_asset in computation.data_assets]))

# %%
results_rm = client.capsules.detach_data_assets(
    capsule_id=os.getenv("CO_CAPSULE_ID"),
    data_assets = ids
)
print(f'Removed all {len(ids)} data assets.')


