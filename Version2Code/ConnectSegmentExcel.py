# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 21:13:54 2023

@author: lenovo
"""

import pandas as pd
import os

input_folder = "D:\\迅雷下载\\KU Leuven\\Yiru Kong - MASTER THESIS\\新喝水组\\分析结果\\NEW POINT WISE EVALUATION"
output_file = "D:\\迅雷下载\\KU Leuven\\Yiru Kong - MASTER THESIS\\新喝水组\\分析结果\\NEW POINT WISE EVALUATION\\SegementResults.xlsx"

dfs = []

for i in range(1, 21):
    file_path = os.path.join(input_folder, f"segementResult{i}.xlsx")
    temp_df = pd.read_excel(file_path, engine='openpyxl')
    temp_df['File'] = f"segementResult{i}"  # 添加一列来存储文件名，以便识别数据来源
    dfs.append(temp_df)

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_excel(output_file, index=False, engine='openpyxl')
