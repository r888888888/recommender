import os
import pandas as pd
import pandas_gbq
import numpy as np
from scipy.sparse import coo_matrix, save_npz
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv
import pickle
from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import NMSLibAlternatingLeastSquares
import signal
import calendar
from pathlib import Path
import datetime
load_dotenv(find_dotenv())

GBQ_PROJECT_ID = "danbooru-1343"
GBQ_TABLE = "danbooru_production.post_votes"
GBQ_KEY_PATH = os.environ.get("GOOGLE_JSON_KEY")
MATRIX_PATH = os.environ.get("MATRIX_PATH")

def query_gbq(year, month):
  start = '{year}-{month:02d}-01 00:00:00'.format(year=year, month=month)
  stop = "{year}-{month:02d}-{eod} 00:00:00".format(year=year, month=month, eod=calendar.monthrange(year, month)[1])
  limit = 1000000
  query = 'SELECT post_id, user_id FROM [{project_id}:{table}] WHERE _PARTITIONTIME >= "{start}" AND _PARTITIONTIME < "{stop}" AND score > 0 LIMIT {limit}'.format(project_id=GBQ_PROJECT_ID, table=GBQ_TABLE, start=start, stop=stop, limit=limit)
  data = pd.read_gbq(query, project_id=GBQ_PROJECT_ID, private_key=GBQ_KEY_PATH)
  try:
    os.makedirs(MATRIX_PATH + "/votes/{year}".format(year=year))
  except os.error:
    None
  data.to_pickle(MATRIX_PATH + "/votes/{year}/{month}.pickle".format(year=year, month=month))

def seed_gbq():
  for month in range(1, 8):
    query_gbq(2018, month)

def train_model():
  model = NMSLibAlternatingLeastSquares(use_gpu=False)
  data = pd.concat([pd.read_pickle(f) for f in Path(MATRIX_PATH).glob("votes/**/*.pickle")])
  data["user_id"] = data["user_id"].astype("category")
  data["post_id"] = data["post_id"].astype("category")
  confidence = 40
  votes_coo = coo_matrix((np.ones(data.shape[0]), (data["post_id"].cat.codes.copy(), data["user_id"].cat.codes.copy()))) * confidence
  votes_csr = votes_coo.tocsr()
  model.fit(votes_coo)
  posts_to_id = {k: v for v, k in enumerate(data["post_id"].cat.categories)}
  ids_to_post = {k: v for v, k in posts_to_id.items()}
  with open(MATRIX_PATH + "/i2p.pickle", "wb") as file:
    pickle.dump(ids_to_post, file)
  with open(MATRIX_PATH + "/p2i.pickle", "wb") as file:
    pickle.dump(posts_to_id, file)
  save_npz(MATRIX_PATH + "/csr", votes_csr)
  with open(MATRIX_PATH + "/model.pickle", "wb") as file:
    pickle.dump(model, file)

now = datetime.datetime.now()
query_gbq(now.year, now.month)
train_model()

pid = int(open("/var/run/recommender/pid", "r").read())
os.kill(pid, signal.SIGHUP)