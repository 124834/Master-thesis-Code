import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from io import StringIO


# output="分析结果/10_output.xlsx"
# theinput="源数据\第J组.xlsx"
# group_name="10"
# testratio=0.048247842

# 用于创建滑动窗口数据for时序信号
def sliding_window(data, labels, window_size):
    data_windows = []
    label_windows = []
    for i in range(len(data) - window_size + 1):
        data_windows.append(data[i:i+window_size])
        label_windows.append(labels[i+window_size//2])
    return np.array(data_windows), np.array(label_windows)

def save_test_results_to_excel(test_x, test_y, y_pred, file_name):
    column_names = ["ax", "ay", "az", "gx", "gy", "gz", "H", "y_pred"]
    test_x = test_x.reshape(-1, 6)
    test_y = test_y.reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    print(f"test_x shape: {test_x.shape}")
    print(f"test_y shape: {test_y.shape}")
    print(f"y_pred shape: {y_pred.shape}")
    data = np.hstack((test_x, test_y, y_pred))
    df = pd.DataFrame(data, columns=column_names)
    df.to_excel(file_name, index=False)

def split_data(x, y, test_ratio):
    split_index = int(len(x) * test_ratio)
    test_x, test_y = x[:split_index], y[:split_index]
    train_x, train_y = x[split_index:], y[split_index:]
    return train_x, test_x, train_y, test_y

# 该函数用于读取数据并使用滑动窗口切分数据
def read_data(TheInput,sampling_rate):
    window_size = 2 * sampling_rate
    # 读取数据文件
    df = pd.read_excel(TheInput, engine='openpyxl')
    # 提取标签列
    y = df.pop('H')
    y = y.values
    # 获取特征数据
    x = df.values
    # 使用滑动窗口切分数据
    x_windows, y_windows = sliding_window(x, y, window_size)
    return x_windows, y

# def read_data_simple():
#     df = pd.read_excel("源数据\第一组.xlsx", engine='openpyxl') #change to correct path
#     #x = df.loc[:, ~(df.columns.isin(['label']))]
#     y = df.pop('H')
#     y=y.values
#     x=df.values
#     return x, y

def save_test_results_to_excel(test_x, test_y, y_pred, file_name):
    column_names = ["ax", "ay", "az", "gx", "gy", "gz", "H", "y_pred"]
    test_x = test_x.reshape(-1, 6)
    test_y = test_y.reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    data = np.hstack((test_x, test_y, y_pred))
    df = pd.DataFrame(data, columns=column_names)
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

def save_evaluation_metrics_to_excel(file_path, metrics):
    df = pd.DataFrame(metrics, index=[0])
    df.to_excel(file_path, index=False, engine='openpyxl')

    
def train(sampling_rate,test_ratio,output,myinput,group_name,MetrixSavePath):
        global window_size
        window_size = 2 * sampling_rate
        x, y = read_data(myinput,sampling_rate)
        # 按照给定的比例从输入数据的开头划分测试集和训练集
        train_x, test_x, train_y, test_y = split_data(x, y, test_ratio) 
        print(y)
        test_x1 = test_x[:, 0, :]
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
        batch_size = 16
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
        # print(test_x1);
        # print(np.shape(test_x1));
        print(test_y);
        print(np.shape(test_y));
        print(y_pred)
        print(np.shape(y_pred))
        save_test_results_to_excel(test_x1, test_y, y_pred, output)
              # 显示分类报告和评价指标
        cr = classification_report(y_true, y_pred)
        ic("分类报告", cr)
        ic("准确度", precision_score(y_true, y_pred))
        ic("f1分数", f1_score(y_true, y_pred))
        ic("召回率", recall_score(y_true, y_pred))

        # 绘制混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        title = 'Confusion matrix'
        cmap = plt.cm.Blues
        classes = ['Drinking', 'Non-drinking']
        confusion_df = pd.DataFrame({ 'TP': [tp], 'TN': [tn],'FP': [fp],'FN': [fn] })
        # cr_df = pd.DataFrame(cr).transpose()
        # cr_df.loc['TN'] = tn
        # cr_df.loc['FP'] = fp
        # cr_df.loc['FN'] = fn
        # cr_df.loc['TP'] = tp
        
        # 将数据框保存到Excel文件中
        # cr_df.to_excel('classification_report_and_cm.xlsx')

        plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = 'd'
        thresh = cm.max() / 2.
        cm = np.array([[tp, fn],
               [fp, tn]])
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(f"分析结果/confusion matrix/group_{group_name}_confusion_matrix.png", dpi=200)
        plt.savefig("confusion matrix.png", dpi=200)
        #plt.show()

        # 绘制loss曲线
        plt.figure(figsize=(5, 5))
        plt.plot([i for i in range(len(epoch_loss))], epoch_loss, label="Loss curve")
        plt.xlabel('Epoch')
        plt.ylabel("Loss")
        labels = labels.long()
        plt.legend(loc='best')
        plt.grid()
        plt.savefig("loss.png")
        plt.savefig(f"分析结果/loss plot/group_{group_name}_loss_plot.png", dpi=200)
        #plt.show()
        
        evaluation_metrics = {
        '{group_name}_分类报告':cr,
        'Precision':    precision_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred)}
        print("cr is",cr)
        cr_df = pd.read_csv(StringIO(cr), sep=' {2,}', engine='python')
        result_df = pd.concat([cr_df, confusion_df], axis=0)
# 将DataFrame保存为Excel文件
        
        # Save classification report to excel
        result_df.to_excel(MetrixSavePath, index=True, header=True, startrow=1, sheet_name="report")


    # 将评价指标保存到指定的Excel文件中
        #save_evaluation_metrics_to_excel(MetrixSavePath,evaluation_metrics)

# def main():
#         sampling_rate = int(2 / 0.4)  # 2 seconds window / 0.4 seconds per data point = 5 data points
#         train(sampling_rate,testratio)

# if __name__ == '__main__':
#         main()

for i in range(1, 7): #一次性跑完
    Alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T"]
    theinput=fr"D:\迅雷下载\KU Leuven\Yiru Kong - MASTER THESIS\新喝水组\分析结果\NEW POINT WISE EVALUATION\RawData_updated\ALL_第{Alphabet[i-1]}组.xlsx"
    output_file = fr"D:\迅雷下载\KU Leuven\Yiru Kong - MASTER THESIS\新喝水组\分析结果\NEW POINT WISE EVALUATION\{i}__output.xlsx"
    TestRatioGroup = [0.04845099, 0.052344676, 0.054206873, 0.046419502, 0.047435246, 0.051328932, 0.050313188, 0.046419502, 0.042695107, 0.04845099, 0.054206873, 0.038801422, 0.043541561, 0.051328932, 0.04845099, 0.049466734, 0.054206873, 0.061824953, 0.061824953, 0.052344676]
    
    test_ratio=TestRatioGroup[i-1]
    group_name=i
    sampling_rate = int(2 / 0.4) 
    
    saveMetrics=fr"D:\迅雷下载\KU Leuven\Yiru Kong - MASTER THESIS\新喝水组\分析结果\NEW POINT WISE EVALUATION\pointWiseMetrix_{i}.xlsx"
    train(sampling_rate,test_ratio,output_file,theinput,group_name,saveMetrics)

# i=1
# Alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T"]
# theinput=fr"D:\迅雷下载\KU Leuven\Yiru Kong - MASTER THESIS\新喝水组\分析结果\NEW POINT WISE EVALUATION\RawData_updated\ALL_第{Alphabet[i-1]}组.xlsx"
# output_file = fr"D:\迅雷下载\KU Leuven\Yiru Kong - MASTER THESIS\新喝水组\分析结果\NEW POINT WISE EVALUATION\{i}__output.xlsx"
# TestRatioGroup = [0.04845099, 0.052344676, 0.054206873, 0.046419502, 0.047435246, 0.051328932, 0.050313188, 0.046419502, 0.042695107, 0.04845099, 0.054206873, 0.038801422, 0.043541561, 0.051328932, 0.04845099, 0.049466734, 0.054206873, 0.061824953, 0.061824953, 0.052344676]

# test_ratio=TestRatioGroup[i-1]
# group_name=i
# sampling_rate = int(2 / 0.4) 

# saveMetrics=fr"D:\迅雷下载\KU Leuven\Yiru Kong - MASTER THESIS\新喝水组\分析结果\NEW POINT WISE EVALUATION\pointWiseMetrix_{i}.xlsx"
# train(sampling_rate,test_ratio,output_file,theinput,group_name,saveMetrics)