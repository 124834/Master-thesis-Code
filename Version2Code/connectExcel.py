# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:14:18 2023

@author: lenovo
"""

import pandas as pd

import os

path = 'D:\\迅雷下载\\KU Leuven\\Yiru Kong - MASTER THESIS\\新喝水组\\分析结果'
df_list = []
for i in range(1, 21):
    filename = f"{i}_output.xlsx"
    file_path = os.path.join(path, filename)
    print(file_path)
    df = pd.read_excel(file_path)
    df_list.append(df)


result = pd.concat(df_list)
result.to_excel(r'D:\迅雷下载\KU Leuven\Yiru Kong - MASTER THESIS\新喝水组\分析结果\final_output.xlsx', index=False)

