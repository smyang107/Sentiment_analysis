import torch
import torch.nn as nn
from transformers import BertModel
from torch.autograd import Variable
import math

MODEL_PATH = "./model/chinese_wwm_pytorch"  # 下载地址见 https://github.com/ymcui/Chinese-BERT-wwm
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# 定义模型输入：文本，输出：文本分类)
class TextClassificationModel(nn.Module):

    def __init__(self, num_classes=3):
        super(TextClassificationModel, self).__init__()

        # Bert Encoder
        #BERT 可以作为一个强大的预训练模型，提供句子表示和词向量嵌入的表现能力。BERT 通过一种叫做 “Masked Language Model” 的任务来预训练模型，即将输入语料中的一些词进行遮盖，要求模型对遮盖位置的词进行推理和预测。这训练方式可以让 BERT 模型学习到丰富的上下文语义信息，使其能够更好地理解句子含义。
        self.bert = BertModel.from_pretrained(MODEL_PATH, return_dict=False).to(device)#

        # BiLSTM
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1,
                            batch_first=True, bidirectional=True).to(device)#其他GRU，tramformer

        # TextCNN（残差）
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 256)).to(device)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 256)).to(device)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 256)).to(device)

        # Attention()
        self.attention = nn.Sequential(
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        ).to(device)

        # Output Layer
        self.fc = nn.Linear(96, num_classes).to(device)

    def forward(self, input_ids, attention_mask):
        input_ids = torch.tensor(input_ids).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)

        # BERT Encoding
        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]#torch.Size([64, 50])-->torch.Size([64, 50, 768])

        # BiLSTM
        lstm_output, _ = self.lstm(bert_output)#torch.Size([64, 50, 768])=>torch.Size([64, 50, 256])，hiddden=128*2

        # TextCNN
        cnn_input = lstm_output.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_size*2]  torch.Size([64, 50, 256])->torch.Size([64, 1, 50, 256])

        cnn_output1 = nn.functional.relu(self.conv1(cnn_input)).squeeze(
            3)  # [batch_size, out_channels, seq_len-kernel_size+1]  torch.Size([64, 1, 50, 256])->torch.Size([64,32,1 49])->torch.Size([64, 32, 49])
        cnn_output2 = nn.functional.relu(self.conv2(cnn_input)).squeeze(
            3)  # [batch_size, out_channels, seq_len-kernel_size+1]  torch.Size([64, 1, 50, 256])->torch.Size([64,1, 32, 48])->torch.Size([64, 32, 48])
        cnn_output3 = nn.functional.relu(self.conv3(cnn_input)).squeeze(# torch.Size([64, 1, 50, 256])->torch.Size([64,1, 32, 47])->torch.Size([64, 32, 47])
            3)  # [batch_size, out_channels, seq_len-kernel_size+1]

        # print((torch.squeeze(cnn_output1, dim=3)).size())
        pool_output1 = nn.functional.max_pool1d(cnn_output1, cnn_output1.size(2)).squeeze(#torch.Size([64, 32, 49])->torch.Size([64, 32])
            2)  # [batch_size, out_channels]
        pool_output2 = nn.functional.max_pool1d(cnn_output2, cnn_output2.size(2)).squeeze(#torch.Size([64, 32, 48])->torch.Size([64, 32])
            2)  # [batch_size, out_channels]
        pool_output3 = nn.functional.max_pool1d(cnn_output3, cnn_output3.size(2)).squeeze(
            2)  # [batch_size, out_channels] torch.Size([64, 32, 47])->torch.Size([64, 32])

        cnn_output = torch.cat((pool_output1, pool_output2, pool_output3), dim=1)  # [batch_size, out_channels*3] torch.Size([64, 96])

        # Attention
        attention_output = self.attention(cnn_output)#torch.Size([64, 96])->torch.Size([64, 1])

        #         print(attention_output.size())
        #         print((torch.mul(attention_output, cnn_output)).size())
        # Output Layer
        output = self.fc(cnn_output)#output  torch.Size([64, 96])-->torch.Size([64, 3])[1,0,0],[0,1,0],[0,0,1]
        return output