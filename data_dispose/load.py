import pickle
import pandas as pd
import json
import numpy as np
import re
import itertools
from collections import Counter

data_dir = open('../dataset/Musical_Instruments_5.json')
datas = []
for line in data_dir.readlines():
    dic = json.loads(line)  # Convert JSON formatted data to a dictionary.
    datas.append(dic)
normal = pd.json_normalize(datas)  # Parse a JSON string into a DataFrame.

# Step1: loading raw review datasets...
users_id = []
items_id = []
ratings = []
reviews = []

for line in datas:
    if str(line['reviewerID']) == 'unknown':
        print("unknown user id")
        continue
    if str(line['asin']) == 'unknown':
        print("unkown item id")
        continue
    users_id.append(str(line['reviewerID']))
    items_id.append(str(line['asin']))
    ratings.append(str(line['overall']))
    reviews.append(line['reviewText'])

data_frame = {'user_id': pd.Series(users_id), 'item_id': pd.Series(items_id),
              'ratings': pd.Series(ratings), 'reviews': pd.Series(reviews)}
data = pd.DataFrame(data_frame)  # [['user_id', 'item_id', 'ratings', 'reviews']]
# data['ratings'] = data['ratings'].apply(lambda x: x - 1.0)
del users_id, items_id, ratings, reviews  # recycle


def get_count(data, id):
    ids = set(data[id].tolist())
    return ids


uidList, iidList = get_count(data, 'user_id'), get_count(data, 'item_id')
userNum_all = len(uidList)
itemNum_all = len(iidList)
print("===============Start:all  rawData size======================")
print(f"dataNum: {data.shape[0]}")
print(f"userNum: {userNum_all}")
print(f"itemNum: {itemNum_all}")
print(f"data densiy: {data.shape[0] / float(userNum_all * itemNum_all):.4f}")
print("===============End: rawData size========================")

user2id = dict((uid, i) for (i, uid) in enumerate(uidList))
item2id = dict((iid, i) for (i, iid) in enumerate(iidList))


def numerize(data):   # Convert user and item to numbers.
    uid = list(map(lambda x: user2id[x], data['user_id']))
    iid = list(map(lambda x: item2id[x], data['item_id']))
    data['user_id'] = uid
    data['item_id'] = iid
    return data


data = numerize(data)
data['ratings'] = pd.to_numeric(data['ratings'])
data['ratings'] = data['ratings'].apply(lambda x: x - 1.0)
print("#######################")
tp_rating = data[['user_id', 'item_id', 'ratings']]

n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_test = tp_rating[test_idx]
tp_train = tp_rating[~test_idx]  # 训练——用户,项目,评分
edge = data[~test_idx]  # 训练用——用户,项目,评分,评论
data = data[~test_idx]

# edge.to_csv('../dataset/instrument/instrument_edge.csv', index=False, header=None)  # The frame of edge (8209, 4) includes User, Item, Rating, and Comment.
# tp_train.to_csv('../dataset/instrument/instrument_train.csv', index=False, header=None)  # Training frame (8209, 3) — User, Item, Rating
# tp_test.to_csv('../dataset/instrument/instrument_test.csv', index=False, header=None)  # Testing frame (8209, 3) — User, Item, Rating

user_meta = {}  # For each user in the data, the corresponding items, ratings, and comments.
item_meta = {}  # For each item in the data, the corresponding users, ratings, and comments.

for i in data.values:
    if i[0] in user_meta:
        user_meta[i[0]].append((i[1], float(i[2]), i[3]))
    else:
        user_meta[i[0]] = [(i[1], float(i[2]), i[3])]  # In the data, the user ID, item ID, and the user's comment on the item corresponding to the i-th row.
    if i[1] in item_meta:
        item_meta[i[1]].append((i[0], float(i[2]), i[3]))
    else:
        item_meta[i[1]] = [(i[0], float(i[2]), i[3])]  # In the data, the item ID, user ID, and the user's comment on the item corresponding to the i-th row.


user_rid = {}  # User ID and all the item IDs that the user has interacted with.
item_rid = {}  # Item ID and all the user IDs that have interacted with the item.
user_rid_ra = {}  # User ID and all the ratings of the items that the user has interacted with.
item_rid_ra = {}  # Item ID and all the ratings that the item has received from users.
user_reviews = {}   # User ID and all the comments that the user has provided.
item_reviews = {}  # Item ID and all the comments provided by the corresponding users.


for u in user_meta:
    user_rid[u] = [i[0] for i in user_meta[u]]
    user_rid_ra[u] = [int(i[1]) for i in user_meta[u]]
    user_reviews[u] = [(i[2]) for i in user_meta[u]]
for i in item_meta:
    item_rid[i] = [x[0] for x in item_meta[i]]
    item_rid_ra[i] = [int(x[1]) for x in item_meta[i]]
    item_reviews[i] = [(x[2]) for x in item_meta[i]]

# pickle.dump(user_reviews, open('../dataset/instrument/user_review.pkl', 'wb'))  # User ID and all the comments given by the user.
# pickle.dump(item_reviews, open('../dataset/instrument/item_review.pkl', 'wb'))  # Item ID and all the comments given by the corresponding users.
# pickle.dump(user_rid, open('../dataset/instrument/user_rid.pkl', 'wb'))  # User ID and all the item IDs that the user has interacted with.
# pickle.dump(item_rid, open('../dataset/instrument/item_rid.pkl', 'wb'))  # Item ID and all the user IDs that have interacted with the item.
# pickle.dump(user_rid_ra, open('../dataset/instrument/user_rid_ra.pkl', 'wb'))  # User ID and all the ratings of the items that the user has interacted with.
# pickle.dump(item_rid_ra, open('../dataset/instrument/item_rid_ra.pkl', 'wb'))  # Item ID and all the ratings that the item has received from users.

print("done")


# pickle.dump(batches_train, open('./dataset/instrument/instrument.train.pkl', 'wb'))