import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5, bidirectional=True): # 初始化模型参数
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 嵌入层
        self.lstm = nn.LSTM(embedding_dim,  # 输入维度
                           hidden_dim,  # 隐藏层维度
                           num_layers=num_layers,  # 层数
                           bidirectional=bidirectional,  # 是否双向
                           dropout=dropout if num_layers > 1 else 0, # 是否dropout
                           batch_first=True) # 是否batch_first
        
        # 如果是双向LSTM，需要乘以2
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes) # 全连接层
        self.dropout = nn.Dropout(dropout) # dropout层
        self.sigmoid = nn.Sigmoid()  # 用于多标签分类
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text)) # 嵌入层
        
        # 打包填充序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # 如果是双向LSTM，连接最后的隐藏状态
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
            
        output = self.dropout(hidden)
        # 使用sigmoid激活函数进行多标签分类
        return self.sigmoid(self.fc(output))
