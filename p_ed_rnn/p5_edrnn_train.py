"""
并行编码器解码器
基于GRU构建的多个编码器和解码器，其中解码器的输入是编码器得到隐藏状态
每个编码器的输入包括用户每天行为的统计情况和每天行为序列的隐藏状态
"""

import torch
import torch.nn as nn
import os
import time
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


class Gru_Encoder(nn.Module):
    """ encoder time series """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(Gru_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0, bidirectional=bidirectional)

    def forward(self, e_input, h0):
        # output: batch_size * L * hidden_size
        # hn: 1 * batch_size * hidden_size
        output, hn = self.gru(e_input, h0)
        return output, hn


class Gru_Decoder(nn.Module):
    """ decoder, input is hidden state of encoder """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(Gru_Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0, bidirectional=bidirectional)

    def forward(self, d_input, h0):
        # output: batch_size * L * hidden_size
        # hn: 1 * batch_size * hidden_size
        output, hn = self.gru(d_input, h0)
        return output, hn


class Rnn_Sequence(nn.Module):

    def __init__(self, seq_input, seq_hidden, seq_rnn_layer, bi, seq_num_embedding, seq_embedding_dim,
                 use_gpu, device):
        super(Rnn_Sequence, self).__init__()
        self.seq_input = seq_input
        self.seq_hidden = seq_hidden
        self.seq_emb = nn.ModuleList()
        self.num_layers = seq_rnn_layer
        # self.seq_num_embedding = seq_num_embedding
        total_embdim = 0
        for i in range(len(seq_num_embedding)):
            self.seq_emb.append(nn.Embedding(seq_num_embedding[i], seq_embedding_dim[i]))
            total_embdim += seq_embedding_dim[i] - 1

        self.rnn = nn.GRU(seq_input+total_embdim, seq_hidden, num_layers=seq_rnn_layer, batch_first=True,
                          dropout=0, bidirectional=bi)
        self.use_gpu = use_gpu
        self.device = device

    def forward(self, input):
        batch_size = input.shape[0]
        days = input.shape[1]
        # input: batch_size * days * 300 * 2
        embed_input = None
        for i, seq_embed in enumerate(self.seq_emb):
            embed_seq = seq_embed(input[:, :, :, i].long())
            if embed_input is None:
                embed_input = embed_seq
            else:
                embed_input = torch.cat((embed_input, embed_seq), 3)
        seq_len = embed_input.shape[2]
        input_size = embed_input.shape[3]
        seq_h0 = self.init_hidden(batch_size*days, self.seq_hidden)
        # embed_input: batch_size * days * 300 * 2embed_size
        rnn_out, rnn_h = self.rnn(embed_input.view(-1, seq_len, input_size), seq_h0)
        # return: batch_size * days * hidden_size
        return rnn_out[:, -1, :].view(batch_size, days, -1)

    def init_hidden(self, batch_size, hidden_size):
        if self.use_gpu and self.device == 0:
            h0 = torch.zeros(self.num_layers, batch_size, hidden_size).cuda()
        elif self.use_gpu and self.device == 1:
            h0 = torch.zeros(self.num_layers, batch_size, hidden_size).cuda(device=torch.device('cuda:1'))
        else:
            h0 = torch.zeros(self.num_layers, batch_size, hidden_size)
        return h0


class Gru_Encoder_Decoder(nn.Module):

    def __init__(self, e_input_size, e_hidden_size, d_input_size, d_hidden_size, use_gpu, device,
                 num_layers=1, bidirectional=False):
        super(Gru_Encoder_Decoder, self).__init__()
        self.use_gpu = use_gpu
        self.device = device
        self.num_layers = num_layers
        self.e_hidden_size = e_hidden_size
        self.d_hidden_size = d_hidden_size
        self.encoder = Gru_Encoder(e_input_size, e_hidden_size, num_layers, bidirectional)
        self.decoder = Gru_Decoder(d_input_size, d_hidden_size, num_layers, bidirectional)

    def forward(self, input, target_len):
        """
        :param input:           input data (batch_size * days * e_input_size)
        :param target_len:      time steps of output
        :return:
        """

        batch_size = input.size(0)
        final_output = self.init_output(batch_size, target_len)
        e_ho = self.init_hidden(batch_size, self.e_hidden_size)
        d_h0 = self.init_hidden(batch_size, self.d_hidden_size)
        # e_output: batch_size * time_steps * e_hidden_size
        e_output, e_hn = self.encoder(input, e_ho)
        for i in range(target_len):
            # decoder_input: batch_size * 1 * e_hidden_size
            decoder_input = torch.unsqueeze(e_output[:, -1, :], 1)
            d_output, d_hn = self.decoder(decoder_input, d_h0)
            d_h0 = d_hn

            final_output[:, i, :] = d_output[:, -1, :]
        # final_output: batch_size * target_len * d_hidden_size
        return final_output

    def init_output(self, batch_size, target_len):
        if self.use_gpu and self.device == 0:
            output = torch.zeros(batch_size, target_len, self.d_hidden_size).cuda()
        elif self.use_gpu and self.device == 1:
            output = torch.zeros(batch_size, target_len, self.d_hidden_size).cuda(device=torch.device('cuda:1'))
        else:
            output = torch.zeros(batch_size, target_len, self.d_hidden_size)
        return output

    def init_hidden(self, batch_size, hidden_size):
        if self.use_gpu and self.device == 0:
            h0 = torch.zeros(self.num_layers, batch_size, hidden_size).cuda()
        elif self.use_gpu and self.device == 1:
            h0 = torch.zeros(self.num_layers, batch_size, hidden_size).cuda(device=torch.device('cuda:1'))
        else:
            h0 = torch.zeros(self.num_layers, batch_size, hidden_size)
        return h0


class Parallel_ED_RNN(nn.Module):

    def __init__(self,
                 input_size, user_info_size, e_hidden_size, d_hidden_size, parallel_num,
                 mlp_hidden_size, mlp_layer, dp,
                 beha_num_embedding, beha_embedding_dim, num_embeddings, embedding_dim,
                 seq_input, seq_hidden, seq_rnn_layer, seq_bi, seq_num_emb, seq_emb_dim,
                 device, use_gpu, gru_layer=1):
        super(Parallel_ED_RNN, self).__init__()

        self.seq_hidden = seq_hidden
        self.rnn_seq = Rnn_Sequence(seq_input, seq_hidden, seq_rnn_layer, seq_bi, seq_num_emb, seq_emb_dim,
                                    use_gpu, device)

        self.beha_embedding_dim = beha_embedding_dim
        self.beha_emb = nn.Embedding(num_embeddings=beha_num_embedding, embedding_dim=beha_embedding_dim)

        self.e_input_size = input_size + beha_embedding_dim + seq_hidden - 1
        self.d_input_size = e_hidden_size
        self.e_hidden_size = e_hidden_size
        self.d_hidden_size = d_hidden_size
        self.gru_eds = nn.ModuleList()
        self.parallel_num = parallel_num
        for i in range(parallel_num):
            self.gru_eds.append(Gru_Encoder_Decoder(self.e_input_size, self.e_hidden_size,
                                                    self.d_input_size, self.d_hidden_size,
                                                    use_gpu=use_gpu, device=device))
        self.atten_linear = nn.Linear(d_hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)

        self.emb = nn.ModuleList()
        total_embdim = 0
        for i in range(len(num_embeddings)):
            self.emb.append(nn.Embedding(num_embeddings=num_embeddings[i], embedding_dim=embedding_dim[i]))
            total_embdim += embedding_dim[i] - 1

        self.userinfo_size = user_info_size
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_layer = mlp_layer
        self.num_layers = gru_layer
        if mlp_layer > 0:
            self.mlp_model = nn.Sequential(
                nn.Linear(self.d_hidden_size + self.userinfo_size + total_embdim, self.mlp_hidden_size),
                nn.ReLU(), nn.Dropout(dp))
            if self.mlp_layer > 1:
                for i in range(self.mlp_layer - 1):
                    self.mlp_model.add_module("linear{}".format(i), nn.Linear(self.mlp_hidden_size, mlp_hidden_size))
                    self.mlp_model.add_module("active{}".format(i), nn.ReLU())
                    self.mlp_model.add_module("dropout{}".format(i), nn.Dropout(dp))
            self.fc = nn.Linear(self.mlp_hidden_size, 1)
        else:
            self.fc = nn.Linear(self.d_hidden_size + self.userinfo_size + total_embdim, 1)

        self.device = device
        self.use_gpu = use_gpu

    def forward(self, input, seq_input, user_info, target_len):
        """
        :param input:           input data (batch_size * days * 3), 3 dimensions are active, music number, channel
        :param seq_input:       behaviour sequence data (batch_size * days * 100 * features) features include page and action
        :param user_info:       basic information about user (age, gender, vip, topics, device, city)
        :param target_len:      time steps of output
        :return:
        """
        # 行为序列转化为隐藏状态
        # seq_out : batch_size * days * seq_hidden
        seq_out = self.rnn_seq(seq_input)

        # 对input data中的离散变量做embedding
        # channel: batch_size * time_steps
        channel = input[:, :, -1]
        # emb_channel: batch_size * time_steps * embedding_dims
        emb_channel = self.beha_emb(channel.long())

        # 合并序列的隐藏状态和input data，作为ED的最终输入
        # final_input: batch_size * time_steps * embedding_dims+2
        final_input = torch.cat((input[:, :, 0:2], emb_channel, seq_out), 2)

        # gru_eds_output : batch_size * target_len * parallel_num * d_hidden_size
        gru_eds_output = None
        for i, gru_ed in enumerate(self.gru_eds):
            # ed_output : batch_size * target_len * 1 * d_hidden_size
            ed_output = torch.unsqueeze(gru_ed(final_input, target_len), 2)
            if gru_eds_output is None:
                gru_eds_output = ed_output
            else:
                gru_eds_output = torch.cat((gru_eds_output, ed_output), 2)

        # linear attention
        # weights: batch_size * target_len * parallel_num
        batch_size = input.shape[0]
        weights = self.softmax(self.atten_linear(gru_eds_output).view(batch_size, target_len, self.parallel_num))
        # weighted_sum : batch_size * target_len * d_hidden_size
        weighted_sum = \
            torch.matmul(torch.unsqueeze(weights, 2), gru_eds_output).view(batch_size, target_len, self.d_hidden_size)

        # 个人信息嵌入并合并
        # continue_info : batch_size * info_dim
        continue_info = user_info[:, 0:4]
        for i, embed in enumerate(self.emb):
            embedding_info = embed(user_info[:, 4 + i].long())
            continue_info = torch.cat((continue_info, embedding_info), 1)

        final_output = None
        for i in range(target_len):
            # 合并continue_info与weighted_sum
            merge_input = torch.cat((weighted_sum[:, i, :], continue_info), 1)
            if self.mlp_layer > 0:
                fc_input = self.mlp_model(merge_input)
                model_output = torch.sigmoid(self.fc(fc_input))
            else:
                model_output = torch.sigmoid(self.fc(merge_input))
            # model_output: batch_size * 1
            if final_output is None:
                final_output = model_output
            else:
                final_output = torch.cat((final_output, model_output), 1)

            # final_output: batch_size * target_len * 1
        return torch.unsqueeze(final_output, -1)

    def train_model(self, train_dataloader, num_epochs, path, name,
                    learning_rate_decay=0, learning_rate=0.01, a=False, start_epoch=1):

        text_path = os.path.join(path, name + '.txt')
        model_path = os.path.join(path, name + '.pt')
        if a:
            f = open(text_path, 'a+')
            f.write('Reload trained model\r\n')
        else:
            f = open(text_path, 'w+')
            f.write('Model Structure\r\n')
            f.write(str(self) + '\r\n')
        f.close()
        print('Model Structure: ', self)
        print('Start Training ... ')
        if self.use_gpu and self.device == 0:
            print("Let's use GPU 0!")
            self.cuda()

        if self.use_gpu and self.device == 1:
            print("Let's use GPU 1!")
            self.cuda(device=torch.device('cuda:1'))

        criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            model_path = os.path.join(path, name +'epoch'+ str(epoch+start_epoch) + '.pt')
            f = open(text_path, 'a+')
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            if learning_rate_decay != 0:
                if epoch % learning_rate_decay == 0:
                    learning_rate = learning_rate / 2
                    optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
                    f.write('at epoch {} learning_rate is updated to {}\r\n'.format(epoch, learning_rate))
                    print('at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))

            losses, aucs = [], []
            self.train()
            pre_time = time.time()
            for train_data, test_data, info_data, beha_seq in train_dataloader:
                if self.use_gpu and self.device == 0:
                    train_data, test_data, info_data, beha_seq = Variable(train_data.cuda()), Variable(test_data.cuda()),  \
                                                       Variable(info_data.cuda()), Variable(beha_seq.cuda())
                if self.use_gpu and self.device == 1:
                    cuda1 = torch.device('cuda:1')
                    train_data, test_data, info_data, beha_seq = Variable(train_data.cuda(device=cuda1)), Variable(
                        test_data.cuda(device=cuda1)), Variable(info_data.cuda(device=cuda1)), \
                                                            Variable(beha_seq.cuda(device=cuda1))
                if not self.use_gpu:
                    train_data, test_data, info_data, beha_seq = Variable(train_data), Variable(test_data), \
                                                            Variable(info_data), Variable(beha_seq)
                optimizer.zero_grad()

                # predict 30 days
                train_data_30 = train_data.clone()
                beha_seq_30 = beha_seq[:, 0:30, :, :]
                pred_30 = self(train_data_30, beha_seq_30, info_data, 30)
                selected_pred_30, selected_label_30 = self.select_specific_days([0,1,2,6,13,29], pred_30, test_data)

                # predict 14 days
                # days_shift_14 = list(range(1, 17))
                # final_pred_14, final_label_14 = \
                #     self.sliding_predict(days_shift_14, train_data, test_data, info_data, [0, 1, 2, 6, 13], 14)

                # predict 7 days
                # days_shift_7 = list(range(17, 24))
                # final_pred_7, final_label_7 = \
                #     self.sliding_predict(days_shift_7, train_data, test_data, info_data, [0, 1, 2, 6], 7)

                # predict 3 days
                days_shift = list(range(1, 28))
                # days_shift = [3,6,9,12,15,18,21,24,27]
                select_days = [0, 1, 2]
                final_pred_3, final_label_3 = \
                    self.sliding_predict(days_shift, train_data, test_data, beha_seq, info_data, select_days, 3)

                # predict 1 days
                days_shift_1 = list(range(28, 29))
                final_pred_1, final_label_1 = \
                    self.sliding_predict(days_shift_1, train_data, test_data, beha_seq, info_data, [0], 1)

                final_label_all = torch.cat((selected_label_30,  final_label_3, final_label_1), 0)
                final_pred_all = torch.cat((selected_pred_30,  final_pred_3, final_pred_1), 0)
                loss = criterion(final_pred_all, final_label_all)
                # loss = criterion(selected_pred_30, selected_label_30)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                # train_auc = roc_auc_score(selected_label_30.tolist(), selected_pred_30.tolist())
                train_auc = roc_auc_score(final_label_all.tolist(), final_pred_all.tolist())
                aucs.append(train_auc)

            train_loss = np.mean(losses)
            ave_auc = np.mean(aucs)

            a = "Epoch: {} Train loss: {:.6f}, Train auc:{:.6f}, Time is {:.2f} \r\n".format(epoch, train_loss, ave_auc,
                                                                                             time.time() - pre_time)
            print(a)
            f.write(a)
            f.close()
            torch.save(self.state_dict(), model_path)

        final_test_pred = self.predict_model(train_dataloader)
        return final_test_pred

    def predict_model(self, test_dataloader):

        with torch.no_grad():
            self.eval()
            pre_time = time.time()
            final_test_pred = None
            for train_data, test_data, info_data, beha_seq in test_dataloader:
                if self.use_gpu and self.device == 0:
                    train_data, test_data, info_data, beha_seq = Variable(train_data.cuda()), Variable(
                        test_data.cuda()), Variable(info_data.cuda()), Variable(beha_seq.cuda())
                if self.use_gpu and self.device == 1:
                    cuda1 = torch.device('cuda:1')
                    train_data, test_data, info_data, beha_seq = Variable(train_data.cuda(device=cuda1)), \
                                                            Variable(test_data.cuda(device=cuda1)), \
                                                            Variable(info_data.cuda(device=cuda1)), \
                                                            Variable(beha_seq.cuda(device=cuda1))
                if not self.use_gpu:
                    train_data, test_data, info_data, beha_seq = Variable(train_data), Variable(test_data), \
                                                            Variable(info_data), Variable(beha_seq)

                # batch_size * 60 * 1
                batch_size = train_data.shape[0]
                input_data = torch.cat((train_data, test_data), 1)
                test_pred = self(input_data, beha_seq, info_data, 30)
                test_output = torch.zeros(batch_size, 6)

                select_days = [0, 1, 2, 6, 13, 29]
                for i in range(6):
                    test_output[:, i] = torch.squeeze(test_pred)[:, select_days[i]]
                if final_test_pred is None:
                    final_test_pred = test_output
                else:
                    final_test_pred = torch.cat((final_test_pred, test_output), 0)
            print("predict time is {:.2f}".format(time.time() - pre_time))
        return final_test_pred

    def sliding_predict(self, days_shift, train_x, test_x, bahavior_seq, info_data, days, tar_len):
        """
        spliding on train dataset to predict target days
        :param days_shift: a list which means the end indexs of train_data
        :param train_x:
        :param test_x:
        :param bahavior_seq: batch_size * 60 * 300 * 2
        :param info_data:
        :param days:
        :param tar_len:    output length
        :return:
        """
        final_pred, final_label = None, None
        for day_shift in days_shift:
            slid_train_x = torch.cat((train_x, test_x[:, 0:day_shift, :]), 1)
            slid_test_x = test_x[:, day_shift:, :].clone()
            slid_beha_seq = bahavior_seq[:, 0:30+day_shift, :, :]
            pred = self(slid_train_x, slid_beha_seq, info_data, tar_len)
            select_pred, select_label = self.select_specific_days(days, pred, slid_test_x)
            if final_pred is None and final_label is None:
                final_pred = select_pred
                final_label = select_label
            else:
                final_label = torch.cat((final_label, select_label), 0)
                final_pred = torch.cat((final_pred, select_pred), 0)
        return final_pred, final_label

    def select_specific_days(self, days, pred_x, label_x):
        """
        从未来x天中选择特定的几天
        :param days:
        :param pred_x:
        :param label_x:
        :return:
        """
        # shape of pred_x is batch_size * x * 2
        select_pred, select_label = None, None
        for j in days:
            if select_label is None and select_pred is None:
                select_pred = pred_x[:, j, 0]
                select_label = label_x[:, j, 0]
            else:
                select_label = torch.cat((select_label, label_x[:, j, 0]), 0)
                select_pred = torch.cat((select_pred, pred_x[:, j, 0]), 0)
        # shape of returned select_pred is len(days)batch_size
        return select_pred, select_label


def prepare_data(path, behavior_path, seq_path, batch_size=100):
    """

    :param path:
    :param days:1,2,3,7,14,30
    :param batch_size:
    :return:
    """
    df = pd.read_csv(path)

    user_info = df.loc[:, ['gender', 'age', 'is_vip', 'topics']].values
    num_embedding = [df['device'].nunique(), df['city'].nunique()]
    device_to_ix = {device: i for i, device in enumerate(df['device'].unique())}
    device_idx = [device_to_ix[d] for d in df['device']]
    # print(device_to_ix)
    # print(df['device'])
    # print(device_idx)
    city_to_ix = {city: i for i, city in enumerate(df['city'].unique())}
    city_idx = [city_to_ix[d] for d in df['city']]
    user_info = np.column_stack((user_info, device_idx))
    user_info = np.column_stack((user_info, city_idx))
    print(num_embedding)
    user_info = user_info.astype('float32')

    # load train_data and test_data
    data = np.load(behavior_path)
    data = data.astype('float32')
    channel_nums = int(np.max(data[:, :, -1])) + 1
    print(channel_nums)
    train_data = data[:, 0:30, :]
    test_data = data[:, 30:, :]

    # bahavior sequence data
    seq_data = np.load(seq_path)

    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(test_data),
                                  torch.from_numpy(user_info), torch.from_numpy(seq_data[:, :, 0:100, :]))
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    # test_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)

    return train_dataloader, num_embedding, channel_nums


if __name__ == "__main__":

    data_path = "../../data/final_processed_data/equal_select_device_active_with_info.csv"
    behavior_path = "../../data/final_processed_data/equal_select_mixed_behavior.npy"
    seq_path = "../../data/final_processed_data/equal_select_matched_p_a_seq100.npy"

    df = pd.read_csv(data_path)

    train_dataloader, num_emb, channel_nums = prepare_data(data_path, behavior_path, seq_path, batch_size=500)
    structure = '5parallel_p_a100'
    name = 'seq2h31aug'
    model = Parallel_ED_RNN(input_size=3, user_info_size=6, e_hidden_size=64, d_hidden_size=64, parallel_num=5,
                            mlp_hidden_size=32, mlp_layer=1, dp=0.2,
                            num_embeddings=num_emb, embedding_dim=[4, 4], beha_num_embedding=channel_nums, beha_embedding_dim=4,
                            seq_input=2, seq_hidden=2, seq_rnn_layer=1, seq_bi=False, seq_num_emb=[24, 9], seq_emb_dim=[4, 4],
                            device=0, use_gpu=True)
    # model_path = 'results/seq2h31aug5parallel_p_a100epoch6.pt'
    # model.load_state_dict(torch.load(model_path))
    test_pred = model.train_model(train_dataloader, num_epochs=6, path='results', name=name + structure,
                                  learning_rate_decay=4)
    # test_pred = model.train_model(train_dataloader, num_epochs=1, path='results', name='seq2h31aug5eopch2'+structure,
    #                               learning_rate_decay=5, learning_rate=0.005, a=True)
    # model.cuda(device=torch.device('cuda:1'))
    # test_pred = model.predict_model(train_dataloader)
    print(test_pred)
    df2 = pd.DataFrame(test_pred.tolist(),
                       columns=['label_1d', 'label_2d', 'label_3d', 'label_7d', 'label_14d', 'label_30d'])
    df2.insert(0, 'device_id', df['device_id'])
    df2 = df2.round(3)
    df2.to_csv('submission_' + name + structure + '_6epoch.csv', index=False)
