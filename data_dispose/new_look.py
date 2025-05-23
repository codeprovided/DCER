import json
import pickle

import pandas as pd

# yelp_2018 = pd.read_csv('../dataset/yelp/yelp_2018.csv', header=None, names=['user', 'item', 'rating', 'text', 'time'])
# data712 = yelp_2018[(yelp_2018['time'] >= '2018-01-01') & (yelp_2018['time'] <= '2018-06-30')]
# # print(df)
# data7_12 = data712.dropna(axis=0)
#
# da = data7_12.groupby('user').filter(lambda x: (len(x) > 10))   # Filter
# data = da.groupby('user').apply(lambda x: x.sort_values('time', ascending=False))
#
# data.to_csv('../dataset/yelp1-6.csv', index=False, header=False)
# data = pd.read_csv('../dataset/6_new_Ele2013.csv', header=None, names=['user', 'item', 'text', 'rating', 'time'])
data = pd.read_csv('../dataset/ele2013/6_new_Ele2013.csv', header=None, names=['user', 'item', 'text', 'rating', 'time'])
# res = pd.DataFrame(columns=('user', 'item', 'rating', 'text', 'time')) # Initialize an empty dataFrame.
# GroupBy = data.groupby('user')
# user = 0
#
# for num in GroupBy:
#     user = user + 1
#     time_1 = 0
#     time_2 = 0
#     time_3 = 0
#     time_4 = 0
#     time_5 = 0
#     time_6 = 0
#     for row in num[1].itertuples(index=True, name='Pandas'):
#         if (getattr(row, 'time') >= '2018-07-01') & (getattr(row, 'time') <= '2018-07-31'):
#             if time_1 == 1:
#                 continue
#             else:
#                 time_1 = 1
#                 continue
#         if (getattr(row, 'time') >= '2018-08-01') & (getattr(row, 'time') <= '2018-08-31'):
#             if time_2 == 1:
#                 continue
#             else:
#                 time_2 = 1
#                 continue
#         if (getattr(row, 'time') >= '2018-09-01') & (getattr(row, 'time') <= '2018-09-30'):
#             if time_3 == 1:
#                 continue
#             else:
#                 time_3 = 1
#                 continue
#         if (getattr(row, 'time') >= '2018-10-01') & (getattr(row, 'time') <= '2018-10-31'):
#             if time_4 == 1:
#                 continue
#             else:
#                 time_4 = 1
#                 continue
#         if (getattr(row, 'time') >= '2018-11-01') & (getattr(row, 'time') <= '2018-11-30'):
#             if time_5 == 1:
#                 continue
#             else:
#                 time_5 = 1
#                 continue
#         if (getattr(row, 'time') >= '2018-12-01') & (getattr(row, 'time') <= '2018-12-31'):
#             if time_6 == 1:
#                 continue
#             else:
#                 time_6 = 1
#                 continue
#     if time_1 & time_2 & time_3 & time_4 & time_5 & time_6:
#         res = pd.concat([res, num[1]], axis=0)  # Concatenate two DataFrames vertically
#
#
# def get_count(data, id):
#     ids = set(data[id].tolist())
#     return ids
#
#
# uidList, iidList = get_count(res, 'user'), get_count(res, 'item')
# userNum_all = len(uidList)
# itemNum_all = len(iidList)
# # data = df.dropna(axis=0)
# res.to_csv('../dataset/filter_yelp7-12.csv', index=False, header=False)
#
# train = data[(data['time'] >= '2018-07-01') & (data['time'] <= '2018-11-30')]
# test = data[(data['time'] >= '2018-12-01') & (data['time'] <= '2018-12-31')]
# tp_rating = train[['user', 'item', 'rating']]
# tp_test = test[['user', 'item', 'rating']]
# data_1 = yelp_2018[(yelp_2018['time'] >= '2018-01-01') & (yelp_2018['time'] <= '2018-01-31')]
# data_2 = yelp_2018[(yelp_2018['time'] >= '2018-02-01') & (yelp_2018['time'] <= '2018-02-28')]
# data_3 = yelp_2018[(yelp_2018['time'] >= '2018-03-01') & (yelp_2018['time'] <= '2018-03-31')]
# data_4 = yelp_2018[(yelp_2018['time'] >= '2018-04-01') & (yelp_2018['time'] <= '2018-04-30')]
# data_5 = yelp_2018[(yelp_2018['time'] >= '2018-05-01') & (yelp_2018['time'] <= '2018-05-31')]
# data_6 = yelp_2018[(yelp_2018['time'] >= '2018-06-01') & (yelp_2018['time'] <= '2018-06-30')]
# data_7 = yelp_2018[(yelp_2018['time'] >= '2018-07-01') & (yelp_2018['time'] <= '2018-07-31')]
# data_8 = yelp_2018[(yelp_2018['time'] >= '2018-08-01') & (yelp_2018['time'] <= '2018-08-31')]
# data_9 = yelp_2018[(yelp_2018['time'] >= '2018-09-01') & (yelp_2018['time'] <= '2018-09-30')]
# data_10 = yelp_2018[(yelp_2018['time'] >= '2018-10-01') & (yelp_2018['time'] <= '2018-10-31')]
# data_11 = yelp_2018[(yelp_2018['time'] >= '2018-11-01') & (yelp_2018['time'] <= '2018-11-30')]
# data_12 = yelp_2018[(yelp_2018['time'] >= '2018-12-01') & (yelp_2018['time'] <= '2018-12-31')]
# # # print(data_1)
# #
# data_7.to_csv('../dataset/yelp/yelp_2018_1.csv', index=False, header=False)
# data_8.to_csv('../dataset/yelp/yelp_2018_2.csv', index=False, header=False)
# data_9.to_csv('../dataset/yelp/yelp_2018_3.csv', index=False, header=False)
# data_10.to_csv('../dataset/yelp/yelp_2018_4.csv', index=False, header=False)
# data_11.to_csv('../dataset/yelp/yelp_2018_5.csv', index=False, header=False)
# data_12.to_csv('../dataset/yelp/yelp_2018_6.csv', index=False, header=False)
# data_7.to_csv('../dataset/yelp_2018_7.csv', index=False, header=False)
# data_8.to_csv('../dataset/yelp_2018_8.csv', index=False, header=False)
# data = pd.read_csv('../dataset/yelp/new_filter_yelp7-12.csv', header=None, names=['user', 'item', 'rating', 'text', 'time'])
tp_rating = data[(data['time'] >= '2013-01-01') & (data['time'] <= '2013-10-31')]
tp_test = data[(data['time'] >= '2013-11-01') & (data['time'] <= '2018-12-31')]
train = tp_rating[['user', 'item', 'rating']]
test = tp_test[['user', 'item', 'rating']]
#
train.to_csv('../dataset/ele2013/train.csv', index=False, header=False)
test.to_csv('../dataset/ele2013/test.csv', index=False, header=False)

# tmp_ele = data[['user', 'item', 'rating']]
# tmp_yelp = data[['user', 'item', 'rating']]
# tmp1 = data_7[['user', 'item', 'rating']]
# tmp2 = data_8[['user', 'item', 'rating']]
# tmp3 = data_9[['user', 'item', 'rating']]
# tmp4 = data_10[['user', 'item', 'rating']]
# tmp5 = data_11[['user', 'item', 'rating']]
# tmp6 = data_12[['user', 'item', 'rating']]
#
# tmp_ele.to_csv('../dataset/ele2013/tmp_ele.csv', index=False, header=False)
# tmp_yelp.to_csv('../dataset/yelp/tmp_yelp.csv', index=False, header=False)
# tmp1.to_csv('../dataset/yelp/tmp1.csv', index=False, header=False)
# tmp2.to_csv('../dataset/yelp/tmp2.csv', index=False, header=False)
# tmp3.to_csv('../dataset/yelp/tmp3.csv', index=False, header=False)
# tmp4.to_csv('../dataset/yelp/tmp4.csv', index=False, header=False)
# tmp5.to_csv('../dataset/yelp/tmp5.csv', index=False, header=False)
# tmp6.to_csv('../dataset/yelp/tmp6.csv', index=False, header=False)

#
# user_meta = {}  # For each user in the data, the corresponding items, ratings, and comments.
# item_meta = {}  # For each item in the data, the corresponding users, ratings, and comments.
#
# for i in data.values:
#     if i[0] in user_meta:
#         user_meta[i[0]].append((i[1], float(i[2]), i[3]))
#     else:
#        user_meta[i[0]] = [(i[1], float(i[2]), i[3])]  # In the data, the user ID, item ID, and the user's comment on the item corresponding to the i-th row.
#     if i[1] in item_meta:
#         item_meta[i[1]].append((i[0], float(i[2]), i[3]))
#     else:
#         item_meta[i[1]] = [(i[0], float(i[2]), i[3])]  # In the data, the item ID, user ID, and the user's comment on the item corresponding to the \(i\)-th row.
#
#
# user_rid = {}  # User ID and all the Item  IDs that the user has interacted with.
# item_rid = {}  # Item ID and all the Item  IDs that the item has interacted with.
# user_rid_ra = {}  # User ID and all the Item  ratings that the user has interacted with.
# item_rid_ra = {}  # Item ID and all the ratings that the Item  has received from users.
# user_reviews = {}   # User ID and all the comments that the user has provided.
# item_reviews = {}  # Item ID and all the comments provided by the corresponding users.
#
#
# for u in user_meta:
#     user_rid[u] = [i[0] for i in user_meta[u]]
#     user_rid_ra[u] = [int(i[1]) for i in user_meta[u]]
#     user_reviews[u] = [(i[2]) for i in user_meta[u]]
# for i in item_meta:
#     item_rid[i] = [x[0] for x in item_meta[i]]
#     item_rid_ra[i] = [int(x[1]) for x in item_meta[i]]
#     item_reviews[i] = [(x[2]) for x in item_meta[i]]


# pickle.dump(user_reviews, open('../dataset/yelp/user_review_6.pkl', 'wb'))  # User ID and all the comments given by the user.
# pickle.dump(item_reviews, open('../dataset/yelp/y7-12_item_review_1.pkl', 'wb'))  # item ID and all the comments given by the corresponding users.
# pickle.dump(user_rid, open('../dataset/yelp/user_rid_6.pkl', 'wb'))  # User ID and all the item IDs that the user has interacted with.
# pickle.dump(item_rid, open('../dataset/yelp/item_rid.pkl', 'wb'))  # Item ID and all the user IDs that have interacted with the item.
# pickle.dump(user_rid_ra, open('../dataset/yelp/user_rid_ra_6.pkl', 'wb'))  # User ID and all the ratings of the items that the user has interacted with.
# pickle.dump(item_rid_ra, open('../dataset/yelp/item_rid_ra.pkl', 'wb'))  # Item ID and all the ratings that the item has received from users.


# history_u_lists = pickle.load(open('../dataset/yelp/yelp2018_user_rid_1.pkl', 'rb'))
# print(history_u_lists)
