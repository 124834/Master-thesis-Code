
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

output="1_output.xlsx"
# 用于创建滑动窗口数据for时序信号
def sliding_window(data, labels, window_size):
    data_windows = []
    label_windows = []
    for i in range(len(data) - window_size + 1):
        data_windows.append(data[i:i+window_size])
        label_windows.append(labels[i+window_size//2])

    return np.array(data_windows), np.array(label_windows)

# 该函数用于读取数据并使用滑动窗口切分数据
def read_data(sampling_rate):
    window_size = 2 * sampling_rate

    # 读取数据文件
    df = pd.read_excel("源数据\第一组.xlsx", engine='openpyxl')
    # 提取标签列
    y = df.pop('H')
    y = y.values
    # 获取特征数据
    x = df.values

    # 使用滑动窗口切分数据
    x_windows, y_windows = sliding_window(x, y, window_size)
    return x_windows, y_windows

def save_test_x_to_excel(test_x, file_name):
    column_names = ["ax", "ay", "az", "gx", "gy", "gz"]
    df = pd.DataFrame(test_x, columns=column_names)
    df.to_excel(file_name, index=False)

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from icecream import ic

class CustomDataset(Dataset):
    '''
    自定义数据集
    '''

    def __init__(self, data, targets, transform=None):
        self.data = data.reshape(-1, 6, window_size)  # 调整特征维度
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x).reshape(6, window_size)  # 调整为 (通道数, 窗口大小)
        return x, y

# 定义CNN模型，包含两个卷积层和一个全连接层
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义第一个卷积层，输入通道数为6，输出通道数为16，卷积核大小为3
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3)
        # 定义第二个卷积层，输入通道数为16，输出通道数为8，卷积核大小为3
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3)
        # 定义全连接层，输入特征数为 (window_size - 2 * (3 - 1)) * 8，输出特征数为2（1，0 两个喝水状态）
        self.fc1 = nn.Linear(in_features=(window_size - 2 * (3 - 1)) * 8, out_features=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))  # 将上一层的输出通过第二个卷积层，并应用ReLU激活函数
        x = x.view(-1, (window_size - 2 * (3 - 1)) * 8)  # 将卷积层输出的张量展平
        x = self.fc1(x)  # 将展平的张量通过全连接层
        return x
    
def train(sampling_rate):
        global window_size
        window_size = 2 * sampling_rate
        x, y = read_data(sampling_rate)# 读取数据
        # 将数据划分为训练集和测试集，其中测试集占比5%
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.05, random_state=123, stratify=y)

        # 创建训练集和测试集的数据集对象
        train_dataset = CustomDataset(data=train_x,
                                      targets=train_y,
                                      transform=transforms.ToTensor(),
                                      )
        test_dataset = CustomDataset(data=test_x,
                                     targets=test_y,
                                     transform=transforms.ToTensor())

        # 实例化CNN模型
        model = CNN()
        # 定义损失函数为交叉熵损失
        criterion = nn.CrossEntropyLoss()
        # 定义优化器为Adam优化器，学习率为0.001
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 设定训练轮数为100轮
        num_epochs = 100
        # 设定批次大小为32
        batch_size = 200
        # 创建训练集和测试集的数据加载器
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        # 训练模型
        epoch_loss = []
        for epoch in range(num_epochs):
            total_loss = 0  # 保存每个epoch的总loss
            correct = 0  # 每个epoch中计算正确的
            total = 0  # 每个epoch的总数据个数
            for i, (images, labels) in enumerate(train_loader):

                # 梯度清零
                optimizer.zero_grad()

                # 模型预测
                outputs = model(images.float())

                # 计算预测的标签
                _, predicted = torch.max(outputs.data, 1)
                # 累加每个batch中正确的标签
                correct += (predicted == labels).sum()

                # 累加每个batch中数据的个数
                total += labels.size(0)

                # 计算loss
                loss = criterion(outputs, labels)
                # 累加每个batch的loss
                total_loss += loss.item()

                # 反向传播
                loss.backward()
                # 更新模型参数
                optimizer.step()
            # print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
            print('Epoch [%d/%d], Loss: %.4f, Acc: %.4f' % (epoch + 1, num_epochs, total_loss, (correct / total)))
            epoch_loss.append(total_loss)

        # 评估模型
        model.eval()
        correct = 0
        total = 0
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
        
              # 显示分类报告和评价指标
        cr = classification_report(y_true, y_pred)
        ic("分类报告", cr)
        ic("准确度", precision_score(y_true, y_pred))
        ic("f1分数", f1_score(y_true, y_pred))
        ic("召回率", recall_score(y_true, y_pred))

        # 绘制混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        title = 'Confusion matrix'
        cmap = plt.cm.Blues
        classes = ['Drinking', 'Non-drinking']

        plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig("confusion matrix.png", dpi=200)
        plt.show()

        # 绘制loss曲线
        plt.figure(figsize=(5, 5))
        plt.plot([i for i in range(len(epoch_loss))], epoch_loss, label="Loss curve")
        plt.xlabel('Epoch')
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig("loss.png")
        plt.show()

def main():
        sampling_rate = int(2 / 0.4)  # 2 seconds window / 0.4 seconds per data point = 5 data points
        train(sampling_rate)

if __name__ == '__main__':
        main()
