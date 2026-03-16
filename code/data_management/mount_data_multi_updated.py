# %%
from codeocean.data_asset import DataAssetAttachParams
from codeocean import CodeOcean
import pandas as pd
import os
import re
from datetime import datetime


def parseSessionID(file_name):
    parts = re.split(r'[_.]', file_name)

    if len(parts[0]) == 6 and parts[0].isdigit():
        aniID = parts[0]
        date = parts[1]
        dateObj = datetime.strptime(date, "%Y-%m-%d")
        raw_id = file_name
    elif parts[0] in ["behavior", "ecephys"]:
        aniID = parts[1]
        date = parts[2] + "_" + parts[3]
        dateObj = datetime.strptime(date, "%Y-%m-%d_%H-%M-%S")
        raw_id = "_".join(parts[1:])
    else:
        aniID = None
        dateObj = None
        raw_id = None

    return aniID, dateObj, raw_id


client = CodeOcean(
    domain="https://codeocean.allenneuraldynamics.org",
    token=os.getenv("API_SECRET"),
)

# %%
# load and parse data ids
script_dir = os.path.dirname(os.path.abspath(__file__))
datalist_dir = os.path.join(script_dir, "session_assets.csv")

data_df = pd.read_csv(datalist_dir)
data_df = data_df[data_df["session_id"].notna() & (data_df["session_id"] != "")]

# %%
col_to_attach = ["raw_data", "sorted"]

# %%
# Build lists of asset IDs and desired mount names
all_ids = []
all_mounts = []

for curr_col in col_to_attach:
    valid_inds = [
        isinstance(s, str) and 30 < len(s) < 40
        for s in data_df[curr_col].to_list()
    ]

    session_ids = list(data_df.loc[valid_inds, "session_id"].values)
    curr_ids = list(data_df.loc[valid_inds, curr_col].values)
    curr_mounts = [f"{session_id}_{curr_col}" for session_id in session_ids]

    all_ids.extend(curr_ids)
    all_mounts.extend(curr_mounts)

# %%
# Optional manual additions
# all_ids.append("f908dd4d-d7ed-4d52-97cf-ccd0e167c659")
# all_mounts.append("all_behavior")

# %%
# Clean / standardize mount names
all_mounts_new = []
for asset_id, mount in zip(all_ids, all_mounts):
    curr_mount = mount

    if "stan" in mount:
        prefix = mount.split("_model_stan")[0]
        aniID, _, _ = parseSessionID(prefix)
        if aniID is not None:
            curr_mount = f"{aniID}_model_stan"

    all_mounts_new.append(curr_mount)

all_mounts = all_mounts_new

# %%
# Create attach params for computation API
data_assets = [
    DataAssetAttachParams(id=asset_id, mount=mount)
    for asset_id, mount in zip(all_ids, all_mounts)
]

# %%
# Attach to the currently running computation / cloud workstation
computation_id = os.getenv("CO_COMPUTATION_ID")
if not computation_id:
    raise ValueError("CO_COMPUTATION_ID is not set in the environment.")

results = client.computations.attach_data_assets(
    computation_id=computation_id,
    attach_params=data_assets,
)

print(f"Attached {len(data_assets)} data assets to computation {computation_id}.")

for data_asset in results:
    print(f"{data_asset.id} mounted as {data_asset.mount}")