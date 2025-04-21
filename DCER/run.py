import argparse
import math
import os
import pickle
from math import sqrt
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from args import get_parser
from datas import get_data
from UV_Aggregators import UV_Aggregator
from UV_Encoders import UV_Encoder


class DCER(nn.Module):

    def __init__(self, enc_u_history_1, enc_u_history_2, enc_u_history_3, enc_u_history_4,
                 enc_v_history, device):
        super(DCER, self).__init__()
        self.device = device
        self.enc_u_history_1 = enc_u_history_1
        self.enc_u_history_2 = enc_u_history_2
        self.enc_u_history_3 = enc_u_history_3
        self.enc_u_history_4 = enc_u_history_4

        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u_history_1.embed_dim

        # Hidden layer.
        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)

        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        # Use mean squared error for regression problems.
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v, flag, datas, out):  # Input information.

        v_embed = self.enc_v_history.features.weight

        all_c = []  # User curiosity matrix.
        # all_unExp = []
        d_min_list = []
        close_items = []
        time = []
        # When it is the training set, the flag is 0, and when it is the test set, the flag is 1. The only difference is the different input data for each time slice, similar to a sliding window.
        if flag == 0:
            # Time slices 1, 2, 3, 4.
            embeds_u_1 = self.enc_u_history_1(nodes_u, datas.h_u_lists_1, datas.h_ura_lists_1, datas.h_ure_lists_1)
            embeds_u_2 = self.enc_u_history_2(nodes_u, datas.h_u_lists_2, datas.h_ura_lists_2, datas.h_ure_lists_2)
            embeds_u_3 = self.enc_u_history_3(nodes_u, datas.h_u_lists_3, datas.h_ura_lists_3, datas.h_ure_lists_3)
            embeds_u_4 = self.enc_u_history_4(nodes_u, datas.h_u_lists_4, datas.h_ura_lists_4, datas.h_ure_lists_4)

            # Calculate the novelty of the item and the user's curiosity level.
            for i in range(len(nodes_u)):  # For a user-item pair.
                # To calculate the novelty of the item for the user, we need to compare the item with the user's historical interacted items. If the similarity is high, it is not novel. If the similarity is low, the item is novel for the user.
                tmp_list_u = datas.h_u0_lists[int(
                    nodes_u[i])]  # The historical behavior of user u, the set of items interacted with historically.
                new_v = v_embed.data[nodes_v[i]]  # The embedding of the current new item.
                d_min = 9999
                # close_item = -1
                for j in tmp_list_u:  # Find the item most similar to the current item in u's historical set, and the distance d between them.
                    old_v = v_embed.data[j]  # The embedding of the historical item.
                    d = self.dist(new_v, old_v)  # Calculate the Euclidean distance.
                    if d < d_min:
                        d_min = d  # Save the minimum distance.

                d_min_list.append(d_min)

                # Calculate the user's curiosity level by averaging the user representations from adjacent time slices.
                c_u = (self.dist(embeds_u_1.data[i], embeds_u_2.data[i]) + self.dist(embeds_u_2.data[i],
                                                                                     embeds_u_3.data[i]) + self.dist(
                    embeds_u_3.data[i], embeds_u_4.data[i])) / 3
                all_c.append(c_u)
        # For testing.
        else:
        # Time slices 1, 2, 3, 4.
            embeds_u_1 = self.enc_u_history_1(nodes_u, datas.h_u_lists_2, datas.h_ura_lists_2, datas.h_ure_lists_2)
            embeds_u_2 = self.enc_u_history_2(nodes_u, datas.h_u_lists_3, datas.h_ura_lists_3, datas.h_ure_lists_3)
            embeds_u_3 = self.enc_u_history_3(nodes_u, datas.h_u_lists_4, datas.h_ura_lists_4, datas.h_ure_lists_4)
            embeds_u_4 = self.enc_u_history_4(nodes_u, datas.h_u_lists_5, datas.h_ura_lists_5, datas.h_ure_lists_5)

            # Calculate the novelty of the item and the user's curiosity level
            for i in range(len(nodes_u)):
                tmp_list_u = datas.h_u1_lists[int(nodes_u[i])]
                new_v = v_embed.data[nodes_v[i]]
                d_min = 9999
                close_item = -1
                for j in tmp_list_u:
                    old_v = v_embed.data[j]
                    d = self.dist(new_v, old_v)
                    if d < d_min:
                        d_min = d
                        close_item = j

                d_min_list.append(d_min)  # The Shortest Distance between the New Project and the Historical Project (unexp)
                close_items.append(close_item)

                # User Curiosity
                c_u = (self.dist(embeds_u_1.data[i], embeds_u_2.data[i]) + self.dist(embeds_u_2.data[i], embeds_u_3.data[i]) + self.dist(embeds_u_3.data[i], embeds_u_4.data[i]))/3
                all_c.append(c_u)

        sum_0 = math.exp(-1 * 4) + math.exp(-1 * 3) + math.exp(-1 * 2) + math.exp(-1 * 1)
        # Calculate the user's curiosity level by averaging the user representations from adjacent time slices
        embeds_u = embeds_u_1 * math.exp(-1 * 4) / sum_0 + embeds_u_2 * math.exp(-1 * 3) / sum_0 + \
                   embeds_u_3 * math.exp(-1 * 2) / sum_0 + embeds_u_4 * math.exp(-1 * 1) / sum_0

        # Item representation.
        embeds_v = self.enc_v_history(nodes_v, datas.h_v_lists, datas.h_vra_lists, datas.h_vre_lists)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)

        all_unExp = []
        # Scale the novelty value to the range 0-6 based on the properties of the novelty activation function.
        new_d_min_list = d_min_list
        max_d_max = max(d_min_list)
        min_d_min = min(d_min_list)
        interval_0 = max_d_max - min_d_min

        for i in range(len(d_min_list)):
            suo = new_d_min_list[i]
            tmp = (suo - min_d_min) / interval_0
            unExp = (6 * tmp) * math.exp(-(6 * tmp))  # Range 0-6.
            all_unExp.append(unExp)

        # Scale the value of user curiosity to the range 0-1
        new_c = all_c
        maxc = max(all_c)
        minc = min(all_c)
        interval = maxc - minc
        for i in range(len(all_c)):
            suo = new_c[i]
            tmp = (suo - minc) / interval
            new_c[i] = tmp

        list_p = torch.tensor(list(map(lambda m, n: m * n, all_unExp, all_c)))  # Calculate the improved results based on user psychology.
        final_p = list_p.reshape(-1, 1).to(self.device)

        ratings = torch.add(scores, final_p)   # Perform linear combination with the original results.

        if out == 1:   # Based on the explanation of multiple time slices, look up the index of the time slice.
            for i in range(len(close_items)):
                if close_items[i] in datas.h_u_lists_5[int(nodes_u[i])]:
                    time.append(1)
                elif close_items[i] in datas.h_u_lists_4[int(nodes_u[i])]:
                    time.append(2)
                elif close_items[i] in datas.h_u_lists_3[int(nodes_u[i])]:
                    time.append(3)
                elif close_items[i] in datas.h_u_lists_2[int(nodes_u[i])]:
                    time.append(4)
                else: time.append(0)

            return ratings, all_unExp, all_c, time
        # return scores.squeeze()
        else:
            return ratings.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list, tmps):
        scores = self.forward(nodes_u, nodes_v, 0, tmps, 0)
        return self.criterion(scores, labels_list)

    @staticmethod
    def dist(a, b):
        d = torch.sqrt(torch.sum((a-b)**2))
        return d


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae, datas):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()  # Optimize   Zero out the gradients before computing the derivatives
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device), datas)
        loss.backward()  # Optimize   Calculate the gradient for each node
        optimizer.step()  # Optimize  Optimize with a step size
        running_loss += loss.detach()
        if i % 100 == 0:  # Print every 100 steps
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader, datas):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v, 1, datas, 0)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))  # Convert a list to a matrix
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


def generation(model, device, test_loader, datas):
    model.eval()
    tmp_pred = []
    nodes_u = []
    nodes_v = []
    unExp_all = []
    c_all = []
    time_all = []

    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            nodes_u.append(list(test_u.data.cpu().numpy()))
            nodes_v.append(list(test_v.data.cpu().numpy()))
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output, unExp, c, time = model.forward(test_u, test_v, 1, datas, 1)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            unExp_all.append(unExp)
            c_all.append(c)
            time_all.append(time)

    nodes_u = np.array(sum(nodes_u, []))
    nodes_v = np.array(sum(nodes_v, []))
    tmp_pred = np.array(sum(tmp_pred, []))  # Convert a list to a matrix
    tmp_pred.reshape(-1)
    unExp_all = np.array(sum(unExp_all, []))
    c_all = np.array(sum(c_all, []))
    time_all = np.array(sum(time_all, []))

    pop_dict = {}  # Item popularity
    i_history = datas.filter_v_lists  # Item ID and the users who have interacted with the item
    i_sort = sorted(i_history.items(), key=lambda item: len(item[1]), reverse=True)
    top = 1  # Select the top 100 items with the highest popularity.
    for key, value in i_sort:
        if top <= 5:
            pop_dict[key] = [5]
        elif top <= 10:
            pop_dict[key] = [10]
        elif top <= 20:
            pop_dict[key] = [20]
        elif top <= 30:
            pop_dict[key] = [30]
        elif top <= 50:
            pop_dict[key] = [50]
        else:
            pop_dict[key] = [100]
        top = top + 1
        if top > 100:
            break

    results = explain(unExp_all, c_all, pop_dict, tmp_pred, nodes_u, nodes_v, time_all)
    return results


def read_vocab(vocab_path, vocabulary, embedding_dim):
    initW = np.random.uniform(-1.0, 1.0, (len(vocabulary), embedding_dim))
    # initW = np.zeros((len(vocabulary), embedding_dim))
    fp = open(vocab_path, encoding='gb18030', errors='ignore')
    lines = fp.readlines()
    i = 0
    for line in lines:
        # print(line)
        line = line.split(" ", 1)
        word = line[0]
        embed = line[1]
        if word in vocabulary:
            idx = vocabulary[word]  # The index of the word.
            initW[idx] = np.fromstring(embed, dtype='float32', sep=" ")
            i = i + 1
            # print(idx)

    return initW


def explain(unexp, c_u, pop, ratings, node_u, node_v, time):
    c_pivot_h = 0.7
    c_pivot_l = 0.3

    filter_node_u = []
    filter_node_v = []
    filter_ratings = []
    filter_c = []
    filter_unexp = []
    filter_time = []
    text_all = []

    results = {}

    for i in range(len(ratings)):
        if ratings[i] >= 3:
            filter_node_u.append(node_u[i])
            filter_node_v.append(node_v[i])
            filter_ratings.append(ratings[i])
            filter_c.append(c_u[i])
            filter_unexp.append(unexp[i])
            filter_time.append(time[i])

    for i in range(len(filter_node_v)):
        text = ''
        if filter_c[i] < c_pivot_l:
            if filter_unexp[i] < 0.2:
                text += 'You prefer to be recommended familiar items, and this is a similar item to your history.'
            text += 'You consumed a similar item ' + str(filter_time[i]) + ' months ago.'
            if filter_node_v[i] in pop.keys():
                text += 'Item '+str(filter_node_v[i])+' is the top '+str(pop.get(filter_node_v[i]))+' items on the popularity list.'
        elif filter_c[i] > c_pivot_h:
            if filter_unexp[i] > 0.2:
                text += 'You are a person who likes to explore new things, and this is a novel item different from your history.'
            else:
                text += 'You consumed a similar item ' + str(filter_time[i]) + ' months ago.'
            if filter_node_v[i] in pop.keys():
                text += 'Item '+str(filter_node_v[i])+' is the top '+str(pop.get(filter_node_v[i]))+' items on the popularity list.'
        else:
            text += 'You consumed a similar item ' + str(filter_time[i]) + ' months ago.'
            if filter_node_v[i] in pop.keys():
                text += 'Item '+str(filter_node_v[i])+' is the top '+str(pop.get(filter_node_v[i]))+' items on the popularity list.'
        text_all.append(text)

        if filter_node_u[i] in results:  # Append explanation
            results[filter_node_u[i]].append((filter_node_v[i], float(filter_ratings[i])+1, text))
        else:
            results[filter_node_u[i]] = [(filter_node_v[i], float(filter_ratings[i])+1, text)]

    return results


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    SEED = 0
    torch.manual_seed(SEED)  # Set a random seed for the CPU.
    torch.cuda.manual_seed(SEED)  # Set a random seed for the current GPU.
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU，Set a random seed for all GPUs.
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # torch.set_printoptions(threshold=np.inf)

    # Parse arguments.
    parser = get_parser()
    opts = parser.parse_args()

    data = get_data()
    datas = data.parse_args()

    train_u = pickle.load(open('../dataset/ele2013/t/u_train1.pkl', 'rb'))
    train_v = pickle.load(open('../dataset/ele2013/t/i_train1.pkl', 'rb'))
    train_r = pickle.load(open('../dataset/ele2013/t/r_train1.pkl', 'rb'))
    test_u = pickle.load(open('../dataset/ele2013/t/u_test1.pkl', 'rb'))
    test_v = pickle.load(open('../dataset/ele2013/t/i_test1.pkl', 'rb'))
    test_r = pickle.load(open('../dataset/ele2013/t/r_test1.pkl', 'rb'))

    para = pickle.load(open('../dataset/ele2013/t/para.pkl', 'rb'))
    # train_u = pickle.load(open('../dataset/yelp/t/u_train.pkl', 'rb'))
    # train_v = pickle.load(open('../dataset/yelp/t/i_train.pkl', 'rb'))
    # train_r = pickle.load(open('../dataset/yelp/t/r_train.pkl', 'rb'))
    # test_u = pickle.load(open('../dataset/yelp/t/u_test.pkl', 'rb'))
    # test_v = pickle.load(open('../dataset/yelp/t/i_test.pkl', 'rb'))
    # test_r = pickle.load(open('../dataset/yelp/t/r_test.pkl', 'rb'))
    # #
    # para = pickle.load(open('../dataset/yelp/t/para.pkl', 'rb'))
    vocabulary = para['user_vocab']  # Set of words

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))  # Pack the tensor
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch_size, shuffle=True)  # Batch data
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opts.test_batch_size, shuffle=True)


    num_users = datas.h_u_lists_1.__len__()
    num_items = datas.h_v_lists.__len__()
    num_ratings = 5

    u2e = nn.Embedding(num_users, opts.embed_dim).to(device)
    v2e = nn.Embedding(num_items, opts.embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, opts.embed_dim).to(device)

    # if opts.glove:
    initW = read_vocab(opts.glove, vocabulary, opts.word_dim)

    # user feature
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, initW, opts.embed_dim, opts.word_dim, opts.vocab_size_u, opts.num_filters,
                                  opts.filter_sizes, opts.seq_len, cuda=device, uv=True)

    enc_u_history_1 = UV_Encoder(u2e, opts.embed_dim, agg_u_history, cuda=device, uv=True)
    enc_u_history_2 = UV_Encoder(u2e, opts.embed_dim, agg_u_history, cuda=device, uv=True)
    enc_u_history_3 = UV_Encoder(u2e, opts.embed_dim, agg_u_history, cuda=device, uv=True)
    enc_u_history_4 = UV_Encoder(u2e, opts.embed_dim, agg_u_history, cuda=device, uv=True)

    # item feature:
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, initW, opts.embed_dim, opts.word_dim, opts.vocab_size_v, opts.num_filters,
                                  opts.filter_sizes, opts.seq_len, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, opts.embed_dim, agg_v_history, cuda=device, uv=False)

    # model
    dcer = DCER(enc_u_history_1, enc_u_history_2, enc_u_history_3, enc_u_history_4, enc_v_history, device).to(device)
    optimizer = torch.optim.RMSprop(dcer.parameters(), lr=opts.lr, alpha=0.9)  # The parameters passed into the neural network, learning efficiency

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
    loss_list = []

    # dcer_best = DCER(enc_u_history_1, enc_u_history_2, enc_u_history_3, enc_u_history_4, enc_u_history_5,
    #                      enc_v_history, device).to(device)
    # print("load model......")
    # dcer_best.load_state_dict(torch.load('../model/dcer_1.pt'))
    # expected_rmse, mae = test(dcer_best, device, test_loader, datas)
    # print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))
    # results = generation(dcer_best, device, test_loader, datas)
    #
    # f = open('../model/results.txt', mode='w')
    # for key, value in results.items():
    #     f.write(str(key) + str(value))
    #     f.write('\n')
    # f.close()
    # print("done")

    for epoch in range(1, opts.epochs + 1):

        train(dcer, device, train_loader, optimizer, epoch, best_rmse, best_mae, datas)
        expected_rmse, mae = test(dcer, device, test_loader, datas)
        loss_list.append(expected_rmse)

        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
            # print("save model")
            # torch.save(dcer.state_dict(), '../model/dcer_test.pt')
        else:
            endure_count += 1
        print("epoch:", epoch)
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 3:

            break
    plt.plot(loss_list)
    plt.show()


if __name__ == '__main__':
    main()
