import os
import pandas as pd
import pandas_gbq
import numpy as np
from scipy.sparse import coo_matrix, save_npz
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv
import pickle

load_dotenv(find_dotenv())

project_id = "danbooru-1343"
table = "danbooru_production.post_votes"
time = datetime.strftime(datetime.now() - timedelta(days=180), format="%Y-%m-%d 00:00:00")
limit = 5000000
query = 'SELECT post_id, user_id FROM [{}:{}] WHERE _PARTITIONTIME >= "{}" AND score > 0 LIMIT {}'.format(project_id, table, time, limit)
google_json_key_path = os.environ.get("GOOGLE_JSON_KEY")
data = pd.read_gbq(query, project_id=project_id, private_key=google_json_key_path)
data["user_id"] = data["user_id"].astype("category")
data["post_id"] = data["post_id"].astype("category")

confidence = 40
votes = coo_matrix((np.ones(data.shape[0]), (data["post_id"].cat.codes.copy(), data["user_id"].cat.codes.copy()))) * confidence
save_npz(os.environ.get("MATRIX_PATH") + "/base-sparse", votes)
posts_to_id = {k: v for v, k in enumerate(data["post_id"].cat.categories)}
ids_to_post = {k: v for v, k in posts_to_id.items()}
with open(os.environ.get("MATRIX_PATH") + "/p2i.pickle", "wb") as file:
  pickle.dump(posts_to_id, file)
with open(os.environ.get("MATRIX_PATH") + "/i2p.pickle", "wb") as file:
  pickle.dump(ids_to_post, file)
