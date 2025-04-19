import pickle

import pandas as pd
import numpy as np
# data = pd.read_csv('../dataset/filter_Electronics_2013.csv', header=None, names=['user', 'item', 'text', 'rating', 'time'])
# res = pd.DataFrame(columns=('user', 'item', 'text', 'rating', 'time'))  # 初始化一个空的dataframe
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
#         if (getattr(row, 'time') >= '2013-01-01') & (getattr(row, 'time') <= '2013-02-28'):
#             if time_1 == 1:
#                 continue
#             else:
#                 time_1 = 1
#                 continue
#         if (getattr(row, 'time') >= '2013-03-01') & (getattr(row, 'time') <= '2013-04-30'):
#             if time_2 == 1:
#                 continue
#             else:
#                 time_2 = 1
#                 continue
#         if (getattr(row, 'time') >= '2013-05-01') & (getattr(row, 'time') <= '2013-06-30'):
#             if time_3 == 1:
#                 continue
#             else:
#                 time_3 = 1
#                 continue
#         if (getattr(row, 'time') >= '2013-07-01') & (getattr(row, 'time') <= '2013-08-31'):
#             if time_4 == 1:
#                 continue
#             else:
#                 time_4 = 1
#                 continue
#         if (getattr(row, 'time') >= '2013-09-01') & (getattr(row, 'time') <= '2013-10-31'):
#             if time_5 == 1:
#                 continue
#             else:
#                 time_5 = 1
#                 continue
#         if (getattr(row, 'time') >= '2013-11-01') & (getattr(row, 'time') <= '2013-12-31'):
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
# print(f"userNum: {userNum_all}")
# print(f"itemNum: {itemNum_all}")
#
# data1 = res[(res['time'] >= '2013-01-01') & (res['time'] <= '2013-02-28')]
# data2 = res[(res['time'] >= '2013-03-01') & (res['time'] <= '2013-04-30')]
# data3 = res[(res['time'] >= '2013-05-01') & (res['time'] <= '2013-06-30')]
# data4 = res[(res['time'] >= '2013-07-01') & (res['time'] <= '2013-08-31')]
# data5 = res[(res['time'] >= '2013-09-01') & (res['time'] <= '2013-10-31')]
# data6 = res[(res['time'] >= '2013-11-01') & (res['time'] <= '2013-12-31')]
#
# res.to_csv('../dataset/6filter_Ele2013.csv', index=False, header=False)
# data1.to_csv('../dataset/1_Ele2013.csv', index=False, header=False)
# data2.to_csv('../dataset/2_Ele2013.csv', index=False, header=False)
# data3.to_csv('../dataset/3_Ele2013.csv', index=False, header=False)
# data4.to_csv('../dataset/4_Ele2013.csv', index=False, header=False)
# data5.to_csv('../dataset/5_Ele2013.csv', index=False, header=False)
# data6.to_csv('../dataset/6_Ele2013.csv', index=False, header=False)
# data0 = pd.read_csv('../dataset/5_Ele2013.csv', header=None, names=['user', 'item', 'text', 'rating', 'time'])
# df = data0[['user', 'item', 'rating', 'text', 'time']]
# data = pd.read_csv('../dataset/new_filter_yelp7-12.csv', header=None, names=['user', 'item', 'rating', 'text', 'time'])
# data = data[['user', 'item', 'rating']]
# n_ratings = data.shape[0]
# test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
# test_idx = np.zeros(n_ratings, dtype=bool)
# test_idx[test] = True
#
# tp_1 = data[test_idx]
# tp_train = data[~test_idx]
# # da = data1[['user', 'item', 'rating', 'text', 'time']]
# tp_1.to_csv('../dataset/new_train.csv', index=False, header=False)
# tp_train.to_csv('../dataset/new_test.csv', index=False, header=False)
u_train = []  # 训练，用户id
i_train = []  # 训练，项目id
r_train = []  # 训练，评分

u_test = []  # 测试，用户id
i_test = []  # 测试，项目id
r_test = []  # 测试，评分
#
print("train")
f_train = open('../dataset/new_train.csv', "r")
i = 0
for line in f_train:
    i = i + 1
    line = line.split(',')
    u_train.append(int(line[0]))
    i_train.append(int(line[1]))
    r_train.append(float(line[2]))

print("test")
f_test = open('../dataset/new_test.csv', "r")
for line in f_test:
    line = line.split(',')
    u_test.append(int(line[0]))
    i_test.append(int(line[1]))
    r_test.append(float(line[2]))
#
pickle.dump(u_train, open('../dataset/u_train0.pkl', 'wb'))
pickle.dump(i_train, open('../dataset/i_train0.pkl', 'wb'))
pickle.dump(r_train, open('../dataset/r_train0.pkl', 'wb'))
pickle.dump(u_test, open('../dataset/u_test0.pkl', 'wb'))
pickle.dump(i_test, open('../dataset/i_test0.pkl', 'wb'))
pickle.dump(r_test, open('../dataset/r_test0.pkl', 'wb'))

# data = pd.read_csv('../dataset/yelp/filter_yelp7-12.csv', header=None, names=['user', 'item', 'rating', 'text', 'time'])
# test = pd.DataFrame(columns=['user', 'item', 'rating', 'text', 'time'])
# test_0 = pd.DataFrame(columns=['user', 'item', 'rating', 'text', 'time'])
# GroupBy = data.groupby('item')
# for i, group in enumerate(GroupBy, 0):
#     x = group[1].copy()
#     x['item'].replace(group[0], i, inplace=True)
#     test = test.append(x)
# print(test)
# GroupBy = test.groupby('user')
# for i, group in enumerate(GroupBy, 0):
#     x = group[1].copy()
#     x['user'].replace(group[0], i, inplace=True)
#     test_0 = test_0.append(x)
# print(test_0)
#
# test_0.to_csv('../dataset/yelp/new_filter_yelp7-12.csv', index=False, header=False)
