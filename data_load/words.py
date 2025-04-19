import json
import pickle
import numpy as np
import re
import itertools
from collections import Counter

import pandas as pd


def clean_str(string):
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences(u_text, u_w_len):
    padding_word = "<PAD/>"
    review_len = u_w_len
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        padded_u_train = []
        for ri in range(len(u_reviews)):
            sentence = u_reviews[ri]
            if review_len > len(sentence):
                num_padding = review_len - len(sentence)
                new_sentence = sentence + [padding_word] * num_padding
                padded_u_train.append((new_sentence))
            else:
                new_sentence = sentence[:review_len]
                padded_u_train.append((new_sentence))
        # full_empty = ([padding_word] * review_len)
        # padded_u_train.insert(0, full_empty)
        u_text2[i] = padded_u_train
    return u_text2


def build_vocab(sentences1):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    # Mapping from index to word
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}

    # word_counts2 = Counter(itertools.chain(*sentences2))
    # # Mapping from index to word
    # vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    # vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # # Mapping from word to index
    # vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    return [vocabulary1, vocabulary_inv1]


def build_input_data(text, vocabulary):
    """
        根据词汇表将句子映射为数字。
    """
    text2 = {}
    for i in text.keys():
        u_reviews = text[i]
        u = [[vocabulary[word] for word in words] for words in u_reviews]
        text2[i] = u
    return text2


user_reviews = pickle.load(open('../dataset/ele2013/ele2013_user_review.pkl', 'rb'))
# user_rid = pickle.load(open('../dataset/ele2013/u_train1.pkl', 'rb'))
# user_rid_ra = pickle.load(open('../dataset/1_user_rid.pkl', 'rb'))

# item_reviews = pickle.load(open('../dataset/ele2013_item_review.pkl', 'rb'))

u_text = {}
# i_text = {}

tmp = open('../dataset/ele2013/6_new_Ele2013.csv', "r")
i = 0
for line in tmp:
    i = i + 1
    line = line.split(',')
    if int(line[0]) not in u_text:
        u_text[int(line[0])] = []
        for s in user_reviews[int(line[0])]:
            s1 = clean_str(s)
            s1 = s1.split(" ")
            u_text[int(line[0])].append(s1)
    # if int(line[1]) not in i_text:
    #     i_text[int(line[1])] = []
    #     for s in item_reviews[int(line[1])]:
    #         s1 = clean_str(s)
    #         s1 = s1.split(" ")
    #         i_text[int(line[1])].append(s1)

# review_len_u = np.array([len(j) for i in u_text.values() for j in i])
# x = np.sort(review_len_u)
# u_w_len = x[int(0.8 * len(review_len_u)) - 1]
#
# review_len_i = np.array([len(j) for i in i_text.values() for j in i])
# y = np.sort(review_len_i)
# i_w_len = y[int(0.8 * len(review_len_i)) - 1]
#
# print("u_w_len:", u_w_len)
# print("i_w_len:", i_w_len)

u_text = pad_sentences(u_text, 346)  # 统一user的评论字数
print("pad user done")
# i_text = pad_sentences(i_text, 346)  # 统一item的评论字数
# print("pad item done")
#
# user_voc = [xx for x in u_text.values() for xx in x]  # 将user评论展开？ 拼接，不标注用户id
# item_voc = [xx for x in i_text.values() for xx in x]

# vocabulary_user, vocabulary_inv_user = build_vocab(user_voc)
# vocabulary_item, vocabulary_inv_item = build_vocab(item_voc)
# print(len(vocabulary_user))
# print(len(vocabulary_item))

# u_text, i_text = build_input_data_1(u_text, i_text, vocabulary_user, vocabulary_item)  # 将评论中的单词转换为数字
# para = pickle.load(open('../dataset/para.pkl', 'rb'))
# vocabulary_user = para['user_vocab']
# vocabulary_item = para['item_vocab']

# u_text = build_input_data(u_text, vocabulary_user)
# i_text = build_input_data(i_text, vocabulary_item)
# print("done")

# para = {}
# para['user_vocab'] = vocabulary_user
# para['item_vocab'] = vocabulary_item

# pickle.dump(u_text, open('../dataset/6_u_text.pkl', 'wb'))
# pickle.dump(i_text, open('../dataset/i_text.pkl', 'wb'))
# pickle.dump(para, open('../dataset/para.pkl', 'wb'))
