# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:05:41 2023

@author: lenovo
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def sliding_window(data, labels, window_size):
    data_windows = []
    label_windows = []
    for i in range(len(data) - window_size + 1):
        data_windows.append(data[i:i+window_size])
        label_windows.append(labels[i+(window_size//2)-1])
    return np.array(data_windows), np.array(label_windows)

def read_data(TheInput,sampling_rate):
    window_size = 2 * sampling_rate
    # 读取数据文件
    df = pd.read_excel(TheInput, engine='openpyxl')
    # 提取标签列
    y = df.pop('H')
    y = y.values
    print("y:",y)
    # 获取特征数据
    x = df.values
    # 使用滑动窗口切分数据
    x_windows, y_windows = sliding_window(x, y, window_size)
    return x_windows, y

i=1
Alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T"]
theinput=fr"D:\迅雷下载\KU Leuven\Yiru Kong - MASTER THESIS\新喝水组\分析结果\NEW POINT WISE EVALUATION\RawData_updated\ALL_第{Alphabet[i-1]}组.xlsx"
output_file = fr"D:\迅雷下载\KU Leuven\Yiru Kong - MASTER THESIS\新喝水组\分析结果\NEW POINT WISE EVALUATION\{i}__output.xlsx"
sampling_rate = int(2 / 0.4) 
x_windows, y_windows,y=read_data(theinput,sampling_rate)