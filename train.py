from util import load_corpus, stopwords, processing_bert
import pandas as pd
from sklearn.model_selection import train_test_split
from model import TextClassificationModel
import torch.nn as nn
# test1000 or Total
course_name = 'Total'
PATH = f"../data/{course_name}_barrage_snownlp_ROSTCM6.xlsx"

# %%

# 分别加载训练集和测试集
df = pd.read_excel(PATH)
df.head()
df_new = df[['弹幕内容', 'Results']]

# %%

df_new.rename(columns={'弹幕内容': 'text',
                       'Results': 'label'}, inplace=True)
df_new['text'] = df_new['text'].apply(lambda x: processing_bert(str(x)))

# %%

x_train, x_valid_test, y_train, y_valid_test = train_test_split(df_new['text'], df_new['label'], test_size=0.2,
                                                                random_state=1)  # 这里的test_size=0.25代表选择1/4的数据作为测试集
x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test, test_size=0.333, random_state=2)
print("训练数据集",x_train.shape,y_train)
print("验证机数据",x_valid,y_valid)
print("测试数据集",x_test,y_test)

# %%

import pandas as pd

data_train = {'text': list(x_train),
              'label': list(y_train)}
data_valid = {'text': list(x_valid),
              'label': list(y_valid)}
data_test = {'text': list(x_test),
             'label': list(y_test)}

df_train = pd.DataFrame(data_train)
df_valid = pd.DataFrame(data_valid)
df_test = pd.DataFrame(data_test)

# %%

import os
from transformers import BertTokenizer, BertModel
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 在我的电脑上不加这一句, bert模型会报错
MODEL_PATH = "./model/chinese_wwm_pytorch"  # 下载地址见 https://github.com/ymcui/Chinese-BERT-wwm

# 加载
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)  # 分词器
bert = BertModel.from_pretrained(MODEL_PATH)  # 模型

# %%

batch_size = 64
max_length = 60
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
# %%

# 数据集
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, df):
        self.data = df["text"].tolist()
        self.label = df["label"].tolist()

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


# 训练集
train_data = MyDataset(df_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 验证集
valid_data = MyDataset(df_valid)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)  # 分批次

# 测试集
test_data = MyDataset(df_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)  # 分批次

# %%




# %%

# 模型评估
def evaluate(model, data_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (words, labels) in enumerate(data_loader):
            # print(words)
            tokens = tokenizer(words, padding='max_length', max_length=50, truncation=True, add_special_tokens=True)
            input_ids = torch.tensor(tokens["input_ids"]).to(device)
            attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
            labels = torch.LongTensor(labels).to(device)

            output = model(input_ids, attention_mask).to(device)

            _, predicted = torch.max(output.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


# %%

# 实例化模型
model = TextClassificationModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=1e-5)

# 训练模型
NUM_EPOCHS = 30

for epoch in range(NUM_EPOCHS):

    for batch_idx, (words, labels) in enumerate(train_loader):
        # print(words)
        tokens = tokenizer(words, padding='max_length', max_length=50, truncation=True, add_special_tokens=True)
        input_ids = torch.tensor(tokens["input_ids"]).to(device)
        attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
        labels = torch.LongTensor(labels).to(device)

        model.train()

        optimizer.zero_grad()

        output = model(input_ids, attention_mask)

        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHS, batch_idx + 1,
                          len(train_loader), loss.item()))


    # 在验证集上评估
    with torch.no_grad():
        valid_accuracy = evaluate(model, valid_loader)
        train_accuracy = evaluate(model, train_loader)
    print('Epoch [{}/{}], Validation Accuracy: {:.5f}%'.format(
    epoch + 1, NUM_EPOCHS, valid_accuracy))
    print('Epoch [{}/{}], Training Accuracy: {:.5f}%'.format(
            epoch + 1, NUM_EPOCHS, train_accuracy))

    # save model
    model_path = f"./model/Bert_Bilstm_TextCNN_{valid_accuracy}.model"
    torch.save(model, model_path)
    print("saved model: ", model_path)

# %%

test_accuracy = evaluate(model, test_loader)
print('Test Accuracy: {:.2f}%'.format(test_accuracy))

# %%


# %%


