# %%
from aind_data_access_api.document_db import MetadataDbClient
import pandas as pd
import os
import natsort
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
all_old_ids = []

def convert_to_old(session_id):
   _, ani_name, date, time = session_id.split('_')
   date = date[:4] + date[5:7] + date[8:10]
   return 'm'+ani_name+'d'+date


for ani_name in ani_names:
   filter = {"subject.subject_id": ani_name}
   response = docdb_api_client.retrieve_docdb_records(
      filter_query=filter)
   data_asset_ids = [curr_response['external_links']['Code Ocean'][0] for curr_response in response] 
   session_ids = ['behavior_' + curr_response['name'].split('_nwb')[0] for curr_response in response]
   old_ids = [convert_to_old(session_id) for session_id in session_ids]
   all_session_names.extend(session_ids)
   all_session_ids.extend(data_asset_ids)
   all_old_ids.extend(old_ids)

 # %%
# save as CSV

df = pd.DataFrame({
   'session_id': all_session_names,
   'raw_data': all_session_ids,
   'old_id': all_old_ids
})

sort_ind = natsort.index_natsorted(all_session_names)
df = df.iloc[sort_ind].reset_index(drop=True)
real_time = df['session_id'].str.endswith('_12-00-00')
df = df[~real_time].reset_index(drop=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, 'hopkins_session_assets.csv')
df.to_csv(save_dir, index=False)


