# %%
from aind_data_access_api.document_db import MetadataDbClient
import pandas as pd
import os
API_GATEWAY_HOST = "api.allenneuraldynamics.org"
# Default database and collection names are set in the client
# To override the defaults, provide the database and collection
# parameters in the constructor

docdb_api_client = MetadataDbClient(
   host=API_GATEWAY_HOST,
)

ani_names = ['ZS059', 'ZS060', 'ZS061', 'ZS062']
all_session_names = []
all_session_ids = []

for ani_name in ani_names:
   filter = {"subject.subject_id": ani_name}
   response = docdb_api_client.retrieve_docdb_records(
      filter_query=filter)
   data_asset_ids = [curr_response['external_links']['Code Ocean'][0] for curr_response in response]
   session_ids = ['behavior_' + curr_response['name'].split('_nwb')[0] for curr_response in response]
   all_session_names.extend(session_ids)
   all_session_ids.extend(data_asset_ids)

# %%
# save as CSV

df = pd.DataFrame({
   'session_id': all_session_names,
   'raw_data': all_session_ids
})
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, 'hopkins_session_assets.csv')
df.to_csv(save_dir, index=False)


