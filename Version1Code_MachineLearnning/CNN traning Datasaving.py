# coding:utf-8  

import pandas as pd
import torch
from sklearn.model_selection import train_test_split



#%%
def read_data():
    df = pd.read_excel("……", engine='openpyxl') #change to correct path
    #x = df.loc[:, ~(df.columns.isin(['label']))]
    y = df.pop('H')
    y=y.values
    x=df.values
    return x, y

#%%

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from icecream import ic

#%%


class CustomDataset(Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data.reshape(-1, 6, 1) 
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x).reshape(-1, 1)
        return x, y


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=1) # 卷积网络
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.fc1 = nn.Linear(in_features=1 * 8, out_features=2) # 用于线性变换


    def forward(self, x):
        #defines the forward pass of the network for a given input tensor x.
        x = nn.functional.relu(self.conv1(x)) 
        x = nn.functional.relu(self.conv2(x))
        #Reshape the output tensor， (batch_size, 8).
        x = x.view(-1, 1 * 8)
        #fully connected (linear) layer (fc1) to the reshaped tensor.
        x = self.fc1(x) 
        return x

def save_test_x_to_excel(test_x, file_name):
  
    column_names = ["ax", "ay", "az", "gx", "gy", "gz"]
    df = pd.DataFrame(test_x, columns=column_names)
    df.to_excel(file_name, index=False)
    

def train(split_ratio=0.2):
    x, y = read_data()

  
    data_size = len(x)
    train_size = int(data_size * (1 - split_ratio))
    test_size = data_size - train_size

    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    train_dataset = CustomDataset(data=train_x,
                                  targets=train_y,
                                  transform=transforms.ToTensor(),
                                  )
    test_dataset = CustomDataset(data=test_x,
                                 targets=test_y,
                                 transform=transforms.ToTensor())
    save_test_x_to_excel(test_x, "output.xlsx")
    model = CNN()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 100 
    batch_size = 1000 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)


    epoch_loss = []
    for epoch in range(num_epochs):
        total_loss = 0 
        correct = 0
        total = 0 
        for i, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs = model(images.float())

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()

            total += labels.size(0)

            loss = criterion(outputs, labels)

            total_loss += loss.item()


            loss.backward()

            optimizer.step()
        # print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
        print('Epoch [%d/%d], Loss: %.4f, Acc: %.4f' % (epoch + 1, num_epochs, total_loss, (correct / total)))
        epoch_loss.append(total_loss)
    import matplotlib.pyplot as plt
    plt.plot([i for i in range(len(epoch_loss))], epoch_loss,label="Loss curve")
    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig("loss.png")


    model.eval()
    correct = 0
    total = 0
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    y_pred = []
    y_true = []
    for images, labels in test_loader:
        outputs = model(images.float())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the model on the %d test images: %d %%' % (total, 100 * correct / total))


   
    column_names = ["ax", "ay", "az", "gx", "gy", "gz"]
    df = pd.DataFrame(test_x, columns=column_names)
    df['original']= y_true
    df['predict'] = y_pred
    df.to_excel('output.xlsx', index=False)
    
    
    # # 计算评价指标
    # # 显示分类报告
    # cr = classification_report(y_true, y_pred)
    # ic("分类报告", cr)

    # # 分类准确度
    # ic("准确度", precision_score(y_true, y_pred))

    # # 分类f1分数
    # ic("f1分数", f1_score(y_true, y_pred))

    # # 计算召回率
    # ic("召回率", recall_score(y_true, y_pred))


    # import matplotlib.pyplot as plt
    # import numpy as np

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文
    # plt.rcParams['axes.unicode_minus'] = False  # 设置负号显示


    # # 获取混淆矩阵数据
    # cm = confusion_matrix(y_true, y_pred)
    # # 定义标题、颜色、标签等
    # title = 'Confusion matrix'
    # cmap = plt.cm.Blues
    # classes = ['Drinking', 'Non-drinking']

    # # 绘制混淆矩阵
    # plt.figure(figsize=(5, 5))
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    # fmt = 'd'
    # thresh = cm.max() / 2.
    # for i, j in np.ndindex(cm.shape):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment='center',
    #              color='white' if cm[i, j] > thresh else 'black')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.tight_layout()
    # plt.savefig("confusion matrix.png", dpi=200)

    # plt.show()




def main():
    train()





if __name__ == '__main__':
    main()









