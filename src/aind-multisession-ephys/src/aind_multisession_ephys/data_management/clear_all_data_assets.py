# %%
from codeocean import CodeOcean
import os
client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=os.getenv("API_SECRET"))

# %%
computations = client.capsules.list_computations(capsule_id=os.getenv("CO_CAPSULE_ID"))
print(os.getenv("CO_CAPSULE_ID"))

ids = []

for computation in computations:
    if computation.data_assets is not None:
        ids += [data_asset.id for data_asset in computation.data_assets if type(data_asset) is not None]

ids = list(set(ids))

print(f'Found {len(ids)} data assets.')
# %%
results_rm = client.capsules.detach_data_assets(
    capsule_id=os.getenv("CO_CAPSULE_ID"),
    data_assets = ids
)
print(f'Removed all {len(ids)} data assets.')



# %%
