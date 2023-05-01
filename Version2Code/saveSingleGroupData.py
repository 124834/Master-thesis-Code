# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:37:08 2023

@author: lenovo
"""

import pandas as pd
import os

# 读取包含所有数据的 Excel 文件
all_data = pd.read_excel("D:\\迅雷下载\\KU Leuven\\Yiru Kong - MASTER THESIS\\新喝水组\\源数据\\final_output.xlsx")

# 数据分组信息
groups = {
    'A': (285, 10),
    'B': (308, 10),
    'C': (319, 12),
    'D': (273, 8),
    'E': (279, 11),
    'F': (302, 12),
    'G': (296, 11),
    'H': (273, 14),
    'I': (251, 10),
    'J': (285, 9),
    'K': (319, 10),
    'L': (228, 9),
    'M': (256, 12),
    'N': (302, 11),
    'O': (285, 11),
    'P': (291, 8),
    'Q': (319, 9),
    'R': (364, 10),
    'S': (364, 11),
    'T': (308, 1)
}

start_index = 0

# 为每个组创建一个 Excel 文件
for group, (rows, _) in groups.items():
    if group == 'A':
        start_index += rows
        continue

    # 提取当前组的数据
    group_data = all_data.iloc[start_index : start_index + rows]

    # 将当前组的数据移动到最前面
    all_data = pd.concat([group_data, all_data.drop(group_data.index)])

    # 保存当前组的数据到一个新的 Excel 文件
    output_file = f"D:\\迅雷下载\\KU Leuven\\Yiru Kong - MASTER THESIS\\新喝水组\\分析结果\\NEW POINT WISE EVALUATION\\RawData_updated\\第{group}组.xlsx"
    group_data.to_excel(output_file, index=False)

    start_index += rows
