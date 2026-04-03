# %%
from codeocean.data_asset import DataAssetParams, DataAssetSearchParams,DataAssetAttachParams, Source, AWSS3Source
import pandas as pd
import os
os.sys.path.append('/root/capsule/code/beh_ephys_analysis')
from utils.beh_functions import parseSessionID
import os, sys
from codeocean import CodeOcean
client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=os.getenv("API_SECRET"))

# %%
# load and parse data ids
script_dir = os.path.dirname(os.path.abspath(__file__))
datalist_dirs = [os.path.join(script_dir, 'hopkins_session_assets.csv'), os.path.join(script_dir, 'hopkins_FP_session_assets.csv')]
# datalist_dir =  [os.path.join(script_dir, 'hopkins_FP_session_assets.csv')]
dfs = [pd.read_csv(curr_dir) for curr_dir in datalist_dirs]
data_df = pd.concat(dfs, ignore_index=True)
data_df = data_df[data_df['session_id'].notna() & (data_df['session_id'] != "")]
  

# %%
col_to_attach = ['raw_data']

# %%
# Lists of strings for id and mount
all_ids = []
all_mounts = []

for curr_col in col_to_attach:
    valid_inds = [True if isinstance(s, str) and 30 < len(s) < 40 else False for s in data_df[curr_col].to_list()]
    session_ids = list(data_df[valid_inds]['session_id'].values)
    curr_ids = list(data_df[valid_inds][curr_col].values)
    curr_mount = [session_curr + '_' + curr_col for session_curr in session_ids]
    all_ids.extend(curr_ids)
    all_mounts.extend(curr_mount)
#%%
# all_ids = all_ids + ['f908dd4d-d7ed-4d52-97cf-ccd0e167c659']
# all_mounts = all_mounts + ['all_behavior']
model_csv = os.path.join(script_dir, 'hopkins_model_assets.csv')
data_df = pd.read_csv(model_csv)
col_to_attach = ['model_stan']
for curr_col in col_to_attach:
    valid_inds = [True if isinstance(s, str) and 30 < len(s) < 40 else False for s in data_df[curr_col].to_list()]
    animal_ids = list(data_df[valid_inds]['animal_id'].values)
    curr_ids = list(data_df[valid_inds][curr_col].values)
    curr_mount = [str(animal_curr) + '_' + curr_col for animal_curr in animal_ids]
    all_ids.extend(curr_ids)
    all_mounts.extend(curr_mount)

# # Generate the list of DataAssetAttachParams objects
# all_mounts_new = []
# for id, mount in zip(all_ids, all_mounts):
#     curr_mount = mount
#     if 'stan' in mount:
#         lists = mount.split('_model_stan')
#         aniID, dateObj, raw_id = parseSessionID(lists[0])
        
#         curr_mount = f'{aniID}_model_stan'
#     all_mounts_new.append(curr_mount)   
        
# all_mounts = all_mounts_new
data_assets = [DataAssetAttachParams(id, mount) for id, mount in zip(all_ids, all_mounts)]

# %%
computation_id = os.getenv("CO_COMPUTATION_ID")
results = client.computations.attach_data_assets(
    computation_id=computation_id,
    attach_params = data_assets,
)

print(f'Attached {len(all_ids)} data assets.')

for data_asset in results:
    print(f'{data_asset.id} mounted as {data_asset.mount}')

# %%
