# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 21:53:18 2023

@author: lenovo
"""

import pandas as pd

# 更新这里的路径为您实际的文件路径
file_path = "D:/迅雷下载/KU Leuven/Yiru Kong - MASTER THESIS/新喝水组/分析结果/NEW POINT WISE EVALUATION"

# 按照指定顺序列出要合并的文件名
files_to_merge = [20, 11, 6, 18, 16, 14, 17, 8, 3, 15]

# 读取并合并所有Excel文件
merged_data = pd.DataFrame()
for file_number in files_to_merge:
    file_name = f"{file_path}/{file_number}__output.xlsx"
    file_data = pd.read_excel(file_name)
    merged_data = pd.concat([merged_data, file_data], axis=0, ignore_index=True)

# 将合并后的数据保存到新的Excel文件
output_file_name = f"{file_path}/merged_output.xlsx"
merged_data.to_excel(output_file_name, index=False, engine='openpyxl')
print("Excel文件已合并.")
