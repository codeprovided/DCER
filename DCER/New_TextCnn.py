import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, word_dim, vocab_size, filters_num, filter_sizes, seq_len, initW, cuda):
        super(TextCNN, self).__init__()

        self.word_dim = word_dim
        self.vocab_size = vocab_size
        self.filter_sizes = list(map(int, filter_sizes.split(",")))  # Convolutional kernel window size.
        self.out_channel = filters_num  # Number of channels (number of convolutional kernels per group).
        self.seq_len = seq_len  # The length of the review, number of words.
        self.num_filter_sizes = len(self.filter_sizes)  # Number of convolutional kernels.
        self.word_embedding = nn.Embedding(self.vocab_size, self.word_dim)

        self.device = cuda
        self.dropout = nn.Dropout(0.5)

        # l-att
        self.att_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(5, self.word_dim), padding=((5 - 1) // 2, 0)),
            nn.Sigmoid()
        )
        # l-att Subsequent convolution, single convolutional kernel
        # self.convs = nn.Sequential(
        #     nn.Conv2d(1, self.out_channel, kernel_size=(1, self.word_dim)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(self.seq_len, 1), stride=(1, 1))
        # )
        # Multiple convolutional kernels.
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, self.out_channel, kernel_size=(s, self.word_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(self.seq_len - s + 1, 1), stride=(1, 1))
        ) for s in self.filter_sizes])

        self.out_dim = self.out_channel * self.num_filter_sizes

        self.reset_weights(initW)

    def reset_weights(self, initW):
        # torch.nn.init.uniform_(self.word_embedding.weight, -0.1, 0.1)

        self.word_embedding.weight.data.copy_(torch.tensor(initW))
        self.word_embedding.weight.requires_grad = False

        # Initialization of a single convolutional kernel.
        # cnns = [self.convs[0], self.att_conv[0]]
        # for cnn in cnns:
        #     nn.init.uniform_(cnn.weight, -0.1, 0.1)
        #     nn.init.constant_(cnn.bias, 0.1)

        # Initialization of multiple convolutional kernels.
        nn.init.uniform_(self.att_conv[0].weight, -0.1, 0.1)
        nn.init.constant_(self.att_conv[0].bias, 0.1)
        for cnn in self.convs:
            nn.init.uniform_(cnn[0].weight, -0.1, 0.1)
            nn.init.constant_(cnn[0].bias, 0.1)

    def forward(self, inputs):
        inputs = torch.LongTensor(inputs)
        inputs = inputs.to(self.device)

        inputs_emb = self.word_embedding(inputs)  # shape[Number of reviews.(batch_size?)，seq_len, word_dim]
        inputs_emb = inputs_emb.view(-1, 1, self.seq_len, self.word_dim).contiguous()

        att_score = self.att_conv(inputs_emb)
        inputs_emb = inputs_emb.mul(att_score)
        pooled_out = [conv(inputs_emb) for conv in self.convs]
        # pooled_out = self.convs(inputs_emb)

        outputs_0 = torch.cat(pooled_out, 3)
        outputs_0 = self.dropout(outputs_0)
        # outputs_0 = F.dropout(outputs_0, training=self.training)
        outputs = outputs_0.view(-1, self.out_dim)
        del att_score, inputs, inputs_emb, pooled_out, outputs_0
        # outputs = pooled_out.view(-1, self.out_channel)
        # del att_score, inputs, inputs_emb, pooled_out

        return outputs