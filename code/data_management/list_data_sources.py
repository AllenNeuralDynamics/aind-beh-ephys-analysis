# %%
from codeocean import CodeOcean
import os
import pandas as pd
client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=os.getenv("API_SECRET"))

# %%
computation_id = os.getenv("CO_COMPUTATION_ID")
computation = client.computations.get_computation(computation_id=computation_id)
ids = [data_asset.id for data_asset in computation.data_assets]

# %%
sources = []
names = []
external = []
for id in ids:
    data_asset = client.data_assets.get_data_asset(id);
    if data_asset.source_bucket is not None:
        source_curr = data_asset.source_bucket.bucket
        external_curr = data_asset.source_bucket.external
    else:
        source_curr = None
        external_curr = None
    sources.append(source_curr)
    names.append(data_asset.name)
    external.append(external_curr)

# %%
sources_df = pd.DataFrame({'id': ids, 'name': names, 'source-bucket': sources, 'external': external})

# %%
# save as csv
script_dir = os.path.dirname(os.path.abspath(__file__))
output_csv = os.path.join(script_dir, 'session_assets_sources.csv')
sources_df.to_csv(output_csv)


