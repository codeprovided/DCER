import argparse


def get_parser():
    # Training settings
    # Create a parser
    parser = argparse.ArgumentParser(description='model parameters')
    # data
    parser.add_argument(
        '--glove', default='../dataset/new.txt')  # Pre-trained word vector file.
    # Add parameters.
    # Training batch size.
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')  # Embedding dimension for user and item.
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')  # Learning rate.
    # Testing batch size.
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--drop_out', default=0.5, type=int)  # dropout
    parser.add_argument('--word_dim', type=int, default=100, metavar='N', help='word embedding size')  # Word embedding dimension.
    parser.add_argument('--vocab_size_u', type=int, default=40291, metavar='N', help='vocab size')  # Number of words in user reviews.
    parser.add_argument('--vocab_size_v', type=int, default=40291, metavar='N', help='vocab size')  # Number of words in item reviews.
    parser.add_argument('--filter_sizes', default='1,2,3', type=str)  # Three convolutional kernels with window sizes of 1, 2, and 3 respectively.
    parser.add_argument('--num_filters', default=100, type=int)  # Number of channels (number of convolutional kernels per group).
    parser.add_argument('--seq_len', default=346, type=int)  # The length of the review (number of words in the review).
    return parser
