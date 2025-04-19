import argparse


def get_parser():
    # Training settings
    # 创建解析器
    parser = argparse.ArgumentParser(description='model parameters')
    # data
    parser.add_argument(
        '--glove', default='../dataset/new.txt')  # 预训练词向量文件
    # 添加参数
    # 训练批次大小
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')  # user，item的嵌入维度
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')  # 学习率
    # 测试批次大小
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--drop_out', default=0.5, type=int)  # dropout
    parser.add_argument('--word_dim', type=int, default=100, metavar='N', help='word embedding size')  # 词嵌入维度
    parser.add_argument('--vocab_size_u', type=int, default=40291, metavar='N', help='vocab size')  # 用户评论中的单词数
    parser.add_argument('--vocab_size_v', type=int, default=40291, metavar='N', help='vocab size')  # 项目评论中的单词数
    parser.add_argument('--filter_sizes', default='1,2,3', type=str)  # 3个卷积核，窗口大小分别为1、2、3
    parser.add_argument('--num_filters', default=100, type=int)  # 通道数（每组卷积核的个数）
    parser.add_argument('--seq_len', default=346, type=int)  # 评论的长度（评论中的单词数目）
    return parser
