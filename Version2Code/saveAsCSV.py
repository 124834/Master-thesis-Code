# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 19:24:31 2023

@author: lenovo
"""

import pandas as pd
import os

input_folder = fr"D:\迅雷下载\KU Leuven\Yiru Kong - MASTER THESIS\新喝水组\分析结果\NEW POINT WISE EVALUATION"
output_folder = "D:/迅雷下载/KU Leuven/Yiru Kong - MASTER THESIS/新喝水组/分析结果/NEW POINT WISE EVALUATION/csv_output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(1, 21):
    input_file = os.path.join(input_folder, f"{i}__output.xlsx")
    output_file = os.path.join(output_folder, f"{i}__output.csv")
    
    df = pd.read_excel(input_file)
    df.to_csv(output_file, index=False, sep=',')
