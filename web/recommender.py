import os
from dotenv import load_dotenv, find_dotenv
from flask import Flask, jsonify, request
from flask_basicauth import BasicAuth
from implicit.als import AlternatingLeastSquares
import numpy as np
from scipy.sparse import coo_matrix, load_npz
import pickle
import datetime

load_dotenv(find_dotenv())
MATRIX_PATH = os.environ.get("MATRIX_PATH")

def load_model():
  global _last_loaded
  global _votes
  global _votes_csr
  global _model
  global _posts_to_id
  global _ids_to_post
  _votes = load_npz(MATRIX_PATH + "/base-sparse.npz")
  _votes_csr = _votes.tocsr()
  with open(MATRIX_PATH + "/i2p.pickle", "rb") as file:
    _ids_to_post = pickle.load(file)
  with open(MATRIX_PATH + "/p2i.pickle", "rb") as file:
    _posts_to_id = pickle.load(file)
  with open(MATRIX_PATH + "/model.pickle", "rb") as file:
    _model = pickle.load(file)
  _last_loaded = datetime.datetime.now()

app = Flask("recommender")
app.config["BASIC_AUTH_USERNAME"] = "danbooru"
app.config["BASIC_AUTH_PASSWORD"] = os.environ.get("RECOMMENDER_KEY")
basic_auth = BasicAuth(app)
load_model()

@app.after_request
def set_cors(response):
  response.headers["Access-Control-Allow-Origin"] = "*"
  return response

@app.route("/recommend/<int:user_id>")
@basic_auth.required
def recommend(user_id):
  global _votes_csr
  global _model
  global _ids_to_post
  matches = _model.recommend(user_id, _votes_csr)
  matches = [(_ids_to_post[idx], score) for idx, score in matches]
  return jsonify(matches)

@app.route("/similar/<int:post_id>")
@basic_auth.required
def similar(post_id):
  global _model
  global _posts_to_id
  global _ids_to_post
  if not post_id in _posts_to_id:
    return jsonify(error="post not in database"), 404
  matches = _model.similar_items(_posts_to_id[post_id])
  matches = [(_ids_to_post[idx], str(score)) for idx, score in matches]
  return jsonify(matches)

@app.route("/updated")
def updated():
  global _last_loaded
  return jsonify(str(_last_loaded))

if __name__ == "__main__":
  app.run(debug=False, host="0.0.0.0")
