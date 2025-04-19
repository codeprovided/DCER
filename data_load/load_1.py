import pickle
import pandas as pd
import json
import numpy as np
import re
import itertools
from collections import Counter


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


def pad_sentences(u_text, u_len, u_w_len):
    padding_word = "<PAD/>"
    review_num = u_len
    review_len = u_w_len
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        padded_u_train = []
        for ri in range(review_num):
            if ri < len(u_reviews):
                sentence = u_reviews[ri]
                if review_len > len(sentence):
                    num_padding = review_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_u_train.append((new_sentence))
                else:
                    new_sentence = sentence[:review_len]
                    padded_u_train.append((new_sentence))
            else:
                new_sentence = [padding_word] * review_len
                padded_u_train.append((new_sentence))
        full_empty = ([padding_word] * review_len, 0)
        padded_u_train.insert(0, full_empty)
        u_text2[i] = padded_u_train
    return u_text2


def build_vocab(sentences1, sentences2):
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

    word_counts2 = Counter(itertools.chain(*sentences2))
    # Mapping from index to word
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # Mapping from word to index
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
        根据词汇表将句子映射为数字。
    """
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = [[vocabulary_u[word] for word in words] for words in u_reviews]
        u_text2[i] = u
    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = [[vocabulary_i[word] for word in words] for words in i_reviews]
        i_text2[j] = i
    return u_text2, i_text2


u_train = []  # 训练，用户id
i_train = []  # 训练，项目id
r_train = []  # 训练，评分

u_test = []  # 测试，用户id
i_test = []  # 测试，项目id
r_test = []  # 测试，评分

u_text = {}
i_text = {}

user_reviews = pickle.load(open('../dataset/instrument/user_review.pkl', 'rb'))
item_reviews = pickle.load(open('../dataset/instrument/item_review.pkl', 'rb'))

print("train")
f_train = open('../dataset/instrument/instrument_train.csv', "r")
i = 0
for line in f_train:
    i = i + 1
    line = line.split(',')
    u_train.append(int(line[0]))
    i_train.append(int(line[1]))
    r_train.append(float(line[2]))
    if int(line[0]) not in u_text:
        u_text[int(line[0])] = []
        for s in user_reviews[int(line[0])]:
            s1 = clean_str(s)
            s1 = s1.split(" ")
            u_text[int(line[0])].append(s1)

    if int(line[1]) not in i_text:
        i_text[int(line[1])] = []
        for s in item_reviews[int(line[1])]:
            s1 = clean_str(s)
            s1 = s1.split(" ")
            i_text[int(line[1])].append(s1)

print("test")
f_test = open('../dataset/instrument/instrument_test.csv', "r")
for line in f_test:
    line = line.split(',')
    u_test.append(int(line[0]))
    i_test.append(int(line[1]))
    r_test.append(float(line[2]))

review_num_u = np.array([len(x) for x in u_text.values()])
x = np.sort(review_num_u)
u_len = x[int(0.9 * len(review_num_u)) - 1]

review_len_u = np.array([len(j) for i in u_text.values() for j in i])
x = np.sort(review_len_u)
u_w_len = x[int(0.9 * len(review_len_u)) - 1]

review_num_i = np.array([len(x) for x in i_text.values()])
y = np.sort(review_num_i)
i_len = y[int(0.9 * len(review_num_i)) - 1]

review_len_i = np.array([len(j) for i in i_text.values() for j in i])
y = np.sort(review_len_i)
i_w_len = y[int(0.9 * len(review_len_i)) - 1]

print("u_w_len:", u_w_len)
print("i_w_len:", i_w_len)

u_text = pad_sentences(u_text, u_w_len)  # 统一user的评论字数
print("pad user done")
i_text = pad_sentences(i_text, i_w_len)  # 统一item的评论字数
print("pad item done")

user_voc = [xx for x in u_text.values() for xx in x]  # 将user评论展开？ 拼接，不标注用户id
item_voc = [xx for x in i_text.values() for xx in x]

# vocabulary_user将user评论中所有的单词转换为数字，vocabulary_item将item评论中所有的单词转换为数字
vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(user_voc, item_voc)
print(len(vocabulary_user))
print(len(vocabulary_item))

u_text, i_text = build_input_data(u_text, i_text, vocabulary_user, vocabulary_item)  # 将评论中的单词转换为数字
print("done")

para = {}
para['user_vocab'] = vocabulary_user
para['item_vocab'] = vocabulary_item
para['u_text'] = u_text
para['i_text'] = i_text


# pickle.dump(u_train, open('../dataset/instrument/u_train.pkl', 'wb'))
# pickle.dump(i_train, open('../dataset/instrument/i_train.pkl', 'wb'))
# pickle.dump(r_train, open('../dataset/instrument/r_train.pkl', 'wb'))
# pickle.dump(u_test, open('../dataset/instrument/u_test.pkl', 'wb'))
# pickle.dump(i_test, open('../dataset/instrument/i_test.pkl', 'wb'))
# pickle.dump(r_test, open('../dataset/instrument/r_test.pkl', 'wb'))
#
# pickle.dump(u_text, open('../dataset/instrument/u_text.pkl', 'wb'))
# pickle.dump(i_text, open('../dataset/instrument/i_text.pkl', 'wb'))
# pickle.dump(para, open('../dataset/instrument/para.pkl', 'wb'))

print("write done")

