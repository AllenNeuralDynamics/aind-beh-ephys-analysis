# %%
from codeocean.data_asset import DataAssetParams, DataAssetSearchParams,DataAssetAttachParams, Source, AWSS3Source
import pandas as pd
import os, sys
from codeocean import CodeOcean
client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=os.getenv("API_SECRET"))

# %%
# load and parse data ids
script_dir = os.path.dirname(os.path.abspath(__file__))
datalist_dir = os.path.join(script_dir, 'session_assets.csv')
data_df = pd.read_csv(datalist_dir)

# %%
col_to_attach = ['raw_data', 'sorted_curated', 'model_stan']

# %%
# Lists of strings for id and mount
all_ids = []
all_mounts = []

for curr_col in col_to_attach:
    session_ids = list(data_df['session_id'].values)
    curr_ids = list(data_df[curr_col].values)
    curr_mount = [session_curr + '_' + curr_col for session_curr in session_ids]
    all_ids.extend(curr_ids)
    all_mounts.extend(curr_mount)
#%%
all_ids = all_ids + ['f908dd4d-d7ed-4d52-97cf-ccd0e167c659']
all_mounts = all_mounts + ['all_behavior']

# Generate the list of DataAssetAttachParams objects
data_assets = [DataAssetAttachParams(id, mount) for id, mount in zip(all_ids, all_mounts)]
data_assets_id = all_ids

# %%
# Attach the generated list
results = client.capsules.attach_data_assets(
    capsule_id=os.getenv("CO_CAPSULE_ID"),
    attach_params=data_assets,
)
print(f'Attached {len(data_assets_id)} data assets.')

for data_asset in results:
    print(f'{data_asset.id} mounted as {data_asset.mount}')
