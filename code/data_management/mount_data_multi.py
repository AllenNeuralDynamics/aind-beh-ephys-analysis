# %%
from codeocean.data_asset import DataAssetParams, DataAssetSearchParams,DataAssetAttachParams, Source, AWSS3Source
import pandas as pd
import os
# Resolve code/beh_ephys_analysis (the folder containing `utils`) relative to this
# file's location, so imports work no matter where the repo is checked out.
import os
import sys
_anchor = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath(os.getcwd())
while _anchor != os.path.dirname(_anchor):
    _beh_ephys_root = os.path.join(_anchor, "code", "beh_ephys_analysis")
    if os.path.isdir(os.path.join(_beh_ephys_root, "utils")):
        if _beh_ephys_root in sys.path:
            sys.path.remove(_beh_ephys_root)
        sys.path.insert(0, _beh_ephys_root)
        break
    _anchor = os.path.dirname(_anchor)
from utils.beh_functions import parseSessionID
import os, sys
from codeocean import CodeOcean
client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=os.getenv("API_SECRET"))

# %%
# load and parse data ids
script_dir = os.path.dirname(os.path.abspath(__file__))
datalist_dir = os.path.join(script_dir, 'session_assets.csv')
data_df = pd.read_csv(datalist_dir)
data_df = data_df[data_df['session_id'].notna() & (data_df['session_id'] != "")]

# %%
col_to_attach = ['raw_data', 'sorted_curated', 'sorted', 'model_stan']

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
data_to_add = [
            ("c1a35fd0-c3aa-47a8-ba40-288b1e39a86a", "alignment_fix"),
            ("ac7c7961-9178-4bf9-9d66-0a426cf3cc24", "dorsal_edges"),
            ("7bf4aa31-226c-4c9c-835b-ae0da5ff1ce0", "merfish_data"), # updated
            ("299fd5aa-1454-4b7a-968c-1e7d2c570d27", "LC_percentile_meshes"), # updated
            ("83b9beba-325e-4a67-85a9-0e8dd46dff17", "all_tongue_movements_04022026"), # updated
            ("60d862b2-173a-4024-8a45-9714d40ae7e3", "scratch_data"), # updated
            ("fcda6874-91fa-488f-bd97-f30e17dd61c3", "LC_retro") # updated
            ]
all_ids.extend([item[0] for item in data_to_add])
all_mounts.extend([item[1] for item in data_to_add])

# Generate the list of DataAssetAttachParams objects
all_mounts_new = []
for id, mount in zip(all_ids, all_mounts):
    curr_mount = mount
    if 'stan' in mount:
        lists = mount.split('_model_stan')
        aniID, dateObj, raw_id = parseSessionID(lists[0])
        
        curr_mount = f'{aniID}_model_stan'
    all_mounts_new.append(curr_mount)   
        
all_mounts = all_mounts_new
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
