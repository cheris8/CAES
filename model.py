import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic
from torch.autograd import Variable

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, use_bidirectional=False, use_dropout=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=use_bidirectional,
                           dropout=0.5 if use_dropout else 0.)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)
        # seq len * batch * embedd
        # print(output.shape)
        # output = output.max(0)[0]
        # print(output.shape)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]),
                                        dim=1))

        return self.fc(hidden.squeeze(0))
        # return self.fc(self.dropout(output))


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 output_dim, use_dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters,
                      kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def forward(self, x):
        x = x.permute(1, 0)
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)


class LSTM_with_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, use_bidirectional=False, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim // 2,
                           bidirectional=use_bidirectional,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def attention_net(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)

        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)

        # attn_output = self.attention_net(output, hidden)
        attn_output = self.attention(output, hidden)

        return self.fc(attn_output.squeeze(0))

class ATT_LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, use_bidirectional=False, dropout=0.5):
        super(ATT_LSTM, self).__init__()
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        bi = use_bidirectional
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.activation = nn.ReLU()

        # self.embeddings = nn.Embedding(vocab_size, self.embed_dim, padding_idx=1)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=1, bias=False, batch_first=True, bidirectional=bi, dropout=self.dropout_rate)
        self.att_layer = nn.Linear(self.hidden_dim, 1)# * 2 if bi else self.hidden_dim, 1)

        if bi == True:
            self._dim = 2
            self.output_layer = nn.Linear(self.hidden_dim * 2,output_dim)
        else:
            self._dim = 1
            self.output_layer = nn.Linear(self.hidden_dim, output_dim)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim * self._dim)

    def init_hidden(self, batch_size):
        return (torch.randn(self._dim,batch_size,self.hidden_dim).cuda(),torch.randn(self._dim,batch_size,self.hidden_dim).cuda())

    def temp_softmax(self, z, T) :
        z = z / T 
        max_z = torch.max(z) 
        exp_z = torch.exp(z-max_z) 
        sum_exp_z = torch.sum(exp_z)
        y = exp_z / sum_exp_z
        return y

    def forward(self, x):
        embedded = self.embeddings(x)
        batch_size = x.size(0)
        hidden_state = self.init_hidden(batch_size)

        # (seq_len, bz, direction * hidden_size)
        output, (_, _) = self.lstm(embedded)

        if self._dim == 1:
            atts = torch.tanh(self.att_layer(output))
            atts = F.gumbel_softmax(atts, tau=0.1, dim=1)
            output = atts * output
        else:
            nb_token = 0
            output_list = []
            for _ in range(2):
                atts = torch.tanh(self.att_layer(output[:,:,nb_token:nb_token+self.hidden_dim]))
                atts = F.softmax(atts, dim=1)
                output_temp = atts * output[:,:,nb_token:nb_token+self.hidden_dim]
                output_list.append(output_temp)
                nb_token += self.hidden_dim
            output = torch.mean(torch.cat(output_list, dim = 2), dim=0)
        return self.output_layer(output)

class SELF_ATTEN_VEC(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, use_bidirectional=False, dropout=0.5):
        super(SELF_ATTEN_VEC, self).__init__()
        self.embed_dim = embedding_dim
        self.word_hidden_dim = hidden_dim
        self.hidden_dim = hidden_dim
        bi = use_bidirectional
        self.n_layers = n_layers
        if bi == True:
            self._dim = 2
        else:
            self._dim = 1
        if bi == True:
            self._dim = 2
            self.output_layer = nn.Linear(self.hidden_dim * 2, output_dim)
        else:
            self._dim = 1
            self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        vocab_size = vocab_size
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.activation = nn.ReLU()
        self.output_dim = output_dim
        self.nb_word_lstm_layer = 1
        self.hard_normalize = False
        self.word_embeddings = nn.Embedding(vocab_size, self.embed_dim, padding_idx=1)
        self.lstm = nn.LSTM(self.embed_dim, self.word_hidden_dim, num_layers=1, batch_first=True, bidirectional=bi, dropout=self.dropout_rate)
        self.word_att_layer = nn.Linear(self.word_hidden_dim * self._dim, self.word_hidden_dim * self._dim)
        self.W_s1 = nn.Linear(self.word_hidden_dim * self._dim, self.word_hidden_dim * self._dim//2)
        self.W_s2 = nn.Linear(self.word_hidden_dim * self._dim//2, 1)
        self.fc_layer = nn.Linear(self.word_hidden_dim * self._dim, self.word_hidden_dim * self._dim // 2)
        self.label = nn.Linear(self.word_hidden_dim * self._dim // 2, self.output_dim)
        self.m_att_loss = 'celoss'
        self.attention_score = None
        print()

    def attention_net(self, lstm_output):
        attn_score_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = F.softmax(attn_score_matrix, dim=1)
        self.attention_score = attn_score_matrix.permute(0, 2, 1)

        return attn_weight_matrix.permute(0, 2, 1), attn_score_matrix.permute(0, 2, 1)

    def custom_ce_loss(self, attention_score, soft_label):
        attention_weight = -1 * F.log_softmax(attention_score, dim=1)
        soft_label = soft_label.squeeze(-1)

        n, c = attention_weight.size(0), attention_weight.size(1)

        att_loss = 0.
        att_loss = soft_label * attention_weight
        att_loss = att_loss.sum() / n

        return att_loss

    def get_att_loss(self, att_scores, att_labels):
        att_scores = att_scores.squeeze(-1)
        
        if self.m_att_loss == 'klloss':
            att_weights = torch.log_softmax(att_scores, dim=1) # dim check하ㄹ 것  

            if self.hard_normalize:        
                zero_vec = torch.ones_like(att_labels) * -9e15
                zero_vec = zero_vec.to('cuda').float()
                att_labels = torch.where(att_labels.float() > 0, att_labels.float(), zero_vec)
            att_labels = F.softmax(att_labels, dim=1)
            att_loss = F.kl_div(att_weights, att_labels.float(), reduction='batchmean')
        elif self.m_att_loss == 'bceloss':
            att_loss = F.binary_cross_entropy(F.sigmoid(att_scores), att_labels)
        elif self.m_att_loss == 'celoss':
            if self.hard_normalize:
                zero_vec = torch.ones_like(att_labels) * -9e15
                zero_vec = zero_vec.to('cuda').float()
                att_labels = torch.where(att_labels.float() > 0, att_labels.float(), zero_vec)
            att_labels = F.softmax(att_labels, dim=1)
            att_loss = self.custom_ce_loss(att_scores, att_labels)
        else:
            print("self.m_att_loss: ", self.m_att_loss)

        return att_loss
                
    def forward(self, input):
        batch_size = input.size(0)
        input = input.permute(1, 0)
        input = self.word_embeddings(input)
        # h_0 = Variable(torch.zeros(self._dim * self.n_layers, batch_size, self.hidden_dim).cuda())
        # c_0 = Variable(torch.zeros(self._dim * self.n_layers, batch_size, self.hidden_dim).cuda())

        input, (_, _) = self.lstm(input)
        
        attn_weight_vec, attn_score_vec = self.attention_net(input)
        attn_weight_vec = attn_weight_vec.permute(0, 2, 1)
        attn_score_vec = attn_score_vec.permute(0, 2, 1)
        # att_loss = self.get_att_loss(attn_score_vec, att_label)
        
        output = attn_weight_vec * input
        output = torch.sum(output, dim=0)
        output = self.fc_layer(self.dropout(self.activation(output)))

        return self.label(self.dropout(self.activation(output)))

class CNN_LSTM(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
