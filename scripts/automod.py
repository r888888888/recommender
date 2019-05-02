import boto3
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pprint
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import os
from sklearn.model_selection import RandomizedSearchCV

fields = {
  'artist_count': lambda x: get_number(x['artist_count']),
  'comment_count': lambda x: get_number(x['comment_count']),
  'copyright_count': lambda x: get_number(x['copyright_count']),
  'note_count': lambda x: get_number(x['note_count']),
  'fav_count': lambda x: get_number(x['fav_count']),
  'tag_count': lambda x: get_number(x['tag_count']),
  'file_size': lambda x: get_number(x['file_size']),
  'width': lambda x: get_number(x['width']),
  'height': lambda x: get_number(x['height']),
  'uploader_median_score': lambda x: float(get_number(x.get('median_score', {'N': 0}))),
  'artist_identified': lambda x: int(x['artist_identified']['BOOL']),
  'copyright_identified': lambda x: int(x['copyright_identified']['BOOL']),
  'character_identified': lambda x: int(x['character_identified']['BOOL']),
  'translated': lambda x: int(x['translated']['BOOL']),
  'is_safe': lambda x: int(x.get("rating", {'S': 'q'})['S'] == 's'),
  'is_questionable': lambda x: int(x.get("rating", {'S': 'q'})['S'] == 'q'),
  'is_explicit': lambda x: int(x.get("rating", {'S': 'q'})['S'] == 'e'),
}

client = boto3.client('dynamodb', region_name="us-west-1")
X = None
y = None
pp = pprint.PrettyPrinter()
classifier = None

def get_number(obj):
  if "N" in obj:
    return obj["N"]
  if "NULL" in obj:
    return 0

def extractFeatures(items):
  return np.array([[fields[key](x) for key in fields.keys()] for x in items])

def extractResults(items):
  return np.array([int(x['is_approved']['BOOL']) for x in items]).astype(int)

if os.path.isfile("model.pickle"):
  print("Loading pickled model")
  with open("model.pickle", "rb") as file:
    classifier = pickle.load(file)

if classifier is None:
  if os.path.isfile("X.bin"):
    with open("X.bin", "rb") as file:
      X = pickle.load(file)

  if os.path.isfile("y.bin"):
    with open("y.bin", "rb") as file:
      y = pickle.load(file)

  if X is None and y is None:
    has_more = True
    last_evaluated_key = None
    X = np.zeros(shape=(0, len(fields.keys())))
    y = np.array([])
    while has_more:
      if last_evaluated_key:
        results = client.scan(TableName="automod_events_production", ExclusiveStartKey=last_evaluated_key)
      else:
        results = client.scan(TableName="automod_events_production")
      if "LastEvaluatedKey" in results:
        last_evaluated_key = results['LastEvaluatedKey']
      else:
        has_more = False
      features = extractFeatures(results['Items'])
      results = extractResults(results['Items'])
      print("Querying", X.shape)
      X = np.concatenate((X, features))
      y = np.concatenate((y, results))
      if X.shape[0] > 1000000:
        has_more = False
    with open("X.bin", "wb") as file:
      pickle.dump(X, file)
    with open("y.bin", "wb") as file:
      pickle.dump(y, file)

  classifier = RandomForestClassifier()
  # X_train, X_test, y_train, y_test = train_test_split(X, y)

  param_grid = { 
    n_estimators: [10, 100, 150],
    bootstrap: [True, False],
    max_depth: [None, 10, 20, 30, 40, 50],
    min_samples_split: [2, 5, 10],
    min_samples_leaf: [1, 2, 4]
  }

  random_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, n_jobs=-1)
  random_search.fit(X, y)
  print(random_search.best_params_)

  with open("model.pickle", "wb") as file:
    pickle.dump(classifier, file)

# y_pred = classifier.predict(X_test)
# importances = {}
# for i, key in enumerate(fields.keys()):
#   importances[key] = classifier.feature_importances_[i]

# pp.pprint(sorted(importances.items(), key=lambda x: x[1]))

# from sklearn.metrics import classification_report

# print(classification_report(y_test, y_pred))

# scores = cross_val_score(classifier, X_test, y_test, cv=5)
# print(scores)
