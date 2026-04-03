# %%
from codeocean import CodeOcean
import os
client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=os.getenv("API_SECRET"))

# %%
computation_id = os.getenv("CO_COMPUTATION_ID")
computation = client.computations.get_computation(computation_id=computation_id)
ids = [data_asset.id for data_asset in computation.data_assets]
results_rm = client.computations.detach_data_assets(
    computation_id=computation_id,
    data_assets = ids
)
print(f'Removed all {len(ids)} data assets.')
computation = client.computations.get_computation(computation_id=computation_id)
ids = []
if computation.data_assets:
    ids = list({data_asset.id for data_asset in computation.data_assets})
print(f'Now {len(ids)} data assets remain.')
# %%

