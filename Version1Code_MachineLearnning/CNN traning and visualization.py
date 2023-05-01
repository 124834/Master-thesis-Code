# coding:utf-8  

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def read_data():
    df = pd.read_excel("XXXX.xlsx", engine='openpyxl')

    y = df.pop('H')
    y=y.values
    x=df.values

    return x, y


import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from icecream import ic




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
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=1) 
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.fc1 = nn.Linear(in_features=1 * 8, out_features=2) 


    def forward(self, x):

        x = nn.functional.relu(self.conv1(x)) 
        x = nn.functional.relu(self.conv2(x))

        x = x.view(-1, 1 * 8)

        x = self.fc1(x)
        return x


def train():
    x, y = read_data()
    
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.05, random_state=123, stratify=y) 

    train_dataset = CustomDataset(data=train_x,
                                targets=train_y,
                                transform=transforms.ToTensor(),
                                )
    test_dataset = CustomDataset(data=test_x,
                               targets=test_y,
                               transform=transforms.ToTensor())
    model = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 100 
    batch_size = 32 

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

    

    cr = classification_report(y_true, y_pred)
    ic("classification report", cr)


    ic("precision", precision_score(y_true, y_pred))

    ic("f1", f1_score(y_true, y_pred))

    ic("recall", recall_score(y_true, y_pred))


    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False  


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




def main():
    train()





if __name__ == '__main__':
    main()










