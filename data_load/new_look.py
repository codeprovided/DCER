import json
import pickle

import pandas as pd

# yelp_2018 = pd.read_csv('../dataset/yelp/yelp_2018.csv', header=None, names=['user', 'item', 'rating', 'text', 'time'])
# data712 = yelp_2018[(yelp_2018['time'] >= '2018-01-01') & (yelp_2018['time'] <= '2018-06-30')]
# # print(df)
# data7_12 = data712.dropna(axis=0)
#
# da = data7_12.groupby('user').filter(lambda x: (len(x) > 10))  # 过滤
# data = da.groupby('user').apply(lambda x: x.sort_values('time', ascending=False))
#
# data.to_csv('../dataset/yelp1-6.csv', index=False, header=False)
# data = pd.read_csv('../dataset/6_new_Ele2013.csv', header=None, names=['user', 'item', 'text', 'rating', 'time'])
data = pd.read_csv('../dataset/ele2013/6_new_Ele2013.csv', header=None, names=['user', 'item', 'text', 'rating', 'time'])
# res = pd.DataFrame(columns=('user', 'item', 'rating', 'text', 'time'))  # 初始化一个空的dataframe
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
#         res = pd.concat([res, num[1]], axis=0)  # 上下合并两个dataframe
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
# user_meta = {}  # data中每一个用户对应的项目们,评分们，评论们
# item_meta = {}  # data中每一个项目对应的用户们,评分们，评论们
#
# for i in data.values:
#     if i[0] in user_meta:
#         user_meta[i[0]].append((i[1], float(i[2]), i[3]))
#     else:
#         user_meta[i[0]] = [(i[1], float(i[2]), i[3])]  # data中第i行对应的用户id,项目id，用户对项目的评论
#     if i[1] in item_meta:
#         item_meta[i[1]].append((i[0], float(i[2]), i[3]))
#     else:
#         item_meta[i[1]] = [(i[0], float(i[2]), i[3])]  # data中第i行对应的项目id，用户id, 用户对项目的评论
#
# user_rid = {}  # 用户id，及用户交互的所有项目id
# item_rid = {}  # 项目id，及项目交互的所有用户id
# user_rid_ra = {}  # 用户id，及用户交互的所有项目rating
# item_rid_ra = {}  # 项目id，及项目交互的所有项目rating
# user_reviews = {}   # 用户id，及用户给出的所有评论
# item_reviews = {}  # 项目id，及对应用户们给出的所有评论
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


# pickle.dump(user_reviews, open('../dataset/yelp/user_review_6.pkl', 'wb'))  # 用户id，及用户给出的所有评论
# pickle.dump(item_reviews, open('../dataset/yelp/y7-12_item_review_1.pkl', 'wb'))  # 项目id，及对应用户们给出的所有评论
# pickle.dump(user_rid, open('../dataset/yelp/user_rid_6.pkl', 'wb'))  # 用户id，及用户交互的所有项目id
# pickle.dump(item_rid, open('../dataset/yelp/item_rid.pkl', 'wb'))  # 项目id，及项目交互的所有用户id
# pickle.dump(user_rid_ra, open('../dataset/yelp/user_rid_ra_6.pkl', 'wb'))  # 用户id，及用户交互的所有项目rating
# pickle.dump(item_rid_ra, open('../dataset/yelp/item_rid_ra.pkl', 'wb'))  # 项目id，及项目交互的所有项目rating

# history_u_lists = pickle.load(open('../dataset/yelp/yelp2018_user_rid_1.pkl', 'rb'))
# print(history_u_lists)
