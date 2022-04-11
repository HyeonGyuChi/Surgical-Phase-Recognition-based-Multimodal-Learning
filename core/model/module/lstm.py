import torch
import torch.nn.functional as F
from core.config.task_info import task_dict


def generate_lstm(args):
    arch_name = args.model
    
    if arch_name == 'lstm':
        model = LSTM(args)

    return model

class LSTM(torch.nn.Module):
    """
        outputs
        - output : output for all input sequences (usually uses for classification)
        - hidden_state : the hidden_state in last time as next hidden state input (future prediction task)
    """
    def __init__(self, args):
        super(LSTM, self).__init__()

        self.args = args
        self.device = self.args.device
        self.n_features = self.args.input_size
        self.seq_size = self.args.clip_size
        self.n_layers = self.args.n_layer # number of LSTM layers (stacked)
        self.n_hidden = self.args.hidden_size # number of hidden states
        self.linear_dim = self.args.linear_dim
        self.bidirectional = self.args.use_bidirectional
        
    
        self.lstm = torch.nn.LSTM(input_size = self.args.input_size, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True,
                                 bidirectional=self.bidirectional,)
        
        self.linear = torch.nn.Linear(self.n_hidden*self.seq_size*(1+self.bidirectional), self.linear_dim * self.seq_size)
        # self.linear = torch.nn.Linear(self.n_hidden*self.seq_size*(1+self.bidirectional), self.linear_dim)
        self.dropout = torch.nn.Dropout(0.3)
        
        self.classifiers = []
                
        self.Softmax = torch.nn.Softmax(dim=-1)
    
    def set_classifiers(self, n_class_list):
        for n_class in n_class_list:
            # self.classifiers.append(torch.nn.Linear(self.linear_dim, n_class * self.seq_size).to(self.device))        
            self.classifiers.append(torch.nn.Linear(self.linear_dim * self.seq_size, n_class * self.seq_size).to(self.device))        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers*(1+self.bidirectional), batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers*(1+self.bidirectional), batch_size, self.n_hidden)
        hidden_state = hidden_state.to(self.device)
        cell_state = cell_state.to(self.device)
        self.hidden = (hidden_state, cell_state)
    
    def get_feature(self, x, key='kinematic'):
        x = x[key]
        batch_size, seq_size, _ = x.size()
        self.init_hidden(batch_size)
        self.lstm.flatten_parameters()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        
        x = lstm_out.contiguous().view(batch_size,-1)
        
        # if self.training:
        #     x = self.dropout(x)
        
        feat = self.linear(x)
        
        return feat
    
    def forward(self, x):    
        feat = self.get_feature(x)
        
        outputs = []
        
        for ci in range(len(self.classifiers)):
            x = self.classifiers[ci](feat)
            x = x.view(feat.size(0), self.seq_size, -1)
            
            out = self.Softmax(x)
            outputs.append(out)
        
        return outputs

