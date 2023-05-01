# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:22:43 2023

@author: lenovo
"""

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import f1_score
import time

# finish=0
# if(finish==0):
#     for i in range(1, 21):
#         input_path = fr"分析结果\csv_output\{i}_output.csv"
#         output_path = fr"D:/迅雷下载/KU Leuven/Yiru Kong - MASTER THESIS/新喝水组/分析结果/segementResult{i}.xlsx"
#         finish=1
#         break
#     # 在这里使用 input_path 和 output_path 进行需要的操作


# path = r'分析结果\csv_output\1_output.csv'
# output_file = "D:/迅雷下载/KU Leuven/Yiru Kong - MASTER THESIS/新喝水组/分析结果/segementResult1.xlsx"
# file = glob.glob(path)
# dl = []
# dl.append(pd.read_csv(file[0], delimiter=','))
# df = pd.concat(dl)
# data = df.drop(['H', 'y_pred'], axis=1).values
# original = df['H'].values
# predict = df['y_pred'].values
# # real_water_volume = df['RealWaterVolume'].values
# # predicted_water_volume = df['PredictWaterVolume'].values
# plotrange=(0,1000)
# originalcolor="yellow"
# predictcolor="blue"
# original2=original[30:300]
# predict2=predict[30:300]

def plot_data(data, original_or_predict, title, data_range,mycolor):
    data = data[data_range[0]:data_range[1]]
    original_or_predict = original_or_predict[data_range[0]:data_range[1]]
    lengt = len(data)
    fig, axs = plt.subplots(2)
    fig.set_size_inches(24, 8)
    fig.suptitle(title)
    x = np.linspace(0, lengt, lengt)

    plt.tick_params(labelsize=20)
    linew = 2

    a1 = data[:, 0]
    a2 = data[:, 1]
    a3 = data[:, 2]
    axs[0].plot(x, a1, label="$acc-x$", color="orange", linewidth=linew)
    axs[0].plot(x, a2, label="$acc-y$", color="blue", linewidth=linew)
    axs[0].plot(x, a3, label="$acc-z$", color="green", linewidth=linew)
    axs[0].set_ylabel('Acceleration', fontsize=20)

    g1 = data[:, 3]
    g2 = data[:, 4]
    g3 = data[:, 5]
    axs[1].plot(x, g1, label="$gyro-x$", color="black", linewidth=linew)
    axs[1].plot(x, g2, label="$gyro-y$", color="red", linewidth=linew)
    axs[1].plot(x, g3, label="$gyro-z$", color="teal", linewidth=linew)
    axs[1].set_ylabel('Angular velocity', fontsize=20)

    for ax in axs:
        ax.set_xlabel('time (s)', fontsize=20)
        ax.legend(loc='best')
        ax.grid()

        start = None
        end = None

        for i in range(1, len(original_or_predict)):
            if original_or_predict[i-1] == 0 and original_or_predict[i] == 1:
                start = i
            elif original_or_predict[i-1] == 1 and original_or_predict[i] == 0:
                end = i
                if start is not None:
                    ax.add_patch(Rectangle((start, ax.get_ylim()[0]), end - start, ax.get_ylim()[1] - ax.get_ylim()[0], alpha=0.3, color=mycolor))
                    start = None

    plt.show()

def plot_only_gestures(original, predict, title, data_range):
    original = original[data_range[0]:data_range[1]]
    predict = predict[data_range[0]:data_range[1]]
    lengt = len(original)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(24, 4)
    fig.suptitle(title)
    x = np.linspace(0, lengt, lengt)

    plt.tick_params(labelsize=20)
    ax.set_ylim(0, 1.01)
    ax.set_xlim(0, lengt) 
    ax.set_ylabel('Drinking Gesture', fontsize=20)
    ax.set_xlabel('time (s)', fontsize=20)
    ax.grid()

    def add_patches(original_or_predict, color, fill=True, linestyle='-'):
        start = None
        end = None

        for i in range(1, len(original_or_predict)):
            if original_or_predict[i-1] == 0 and original_or_predict[i] == 1:
                start = i
                if i == 1 and original_or_predict[0] == 1:
                    start = 0
            elif original_or_predict[i-1] == 1 and original_or_predict[i] == 0:
                end = i
                if start is not None:
                    if fill:
                        ax.add_patch(Rectangle((start, ax.get_ylim()[0]), end - start, ax.get_ylim()[1] - ax.get_ylim()[0], alpha=0.3, color=color, fill=fill))
                    else:
                        ax.hlines(1, start, end, color=color, linewidth=3)
                    start = None

    # 画 original 的竖直线
    for i in range(lengt):
        if original[i] == 1 and original[i-1]==0 :
            ax.vlines(i, 0, 1, color='black', linewidth=3)
        if original[i] == 0 and original[i-1]==1 :
            ax.vlines(i, 0, 1, color='black', linewidth=3)
    add_patches(original, 'black', fill=False)
    add_patches(predict, 'blue', fill=True)

    # 添加图例
    add_patches(original, 'black', fill=False)
    add_patches(predict, 'blue', fill=True)

    # 添加图例
    ax.plot([], [], color='black', linewidth=3, label='Truth Drinking Gesture')
    ax.add_patch(Rectangle((0, 0), 1, 1, alpha=0.3, color='blue', label='Predicted Drinking Gesture'))
    ax.legend(fontsize=16, loc='best')

    plt.show()

def plot_water_volume(real_water_volume, predicted_water_volume, title, data_range):
    real_water_volume = real_water_volume[data_range[0]:data_range[1]]
    predicted_water_volume = predicted_water_volume[data_range[0]:data_range[1]]
    
    length = len(real_water_volume)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(24, 4)
    fig.suptitle(title)
    
    ax.tick_params(labelsize=20)
    ax.set_xlim(0, length)
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylim(0, max(max(real_water_volume), max(predicted_water_volume)) * 1.1)
    ax.set_ylabel('Water Volume (mL)', fontsize=20)
    ax.grid()

    # Add lines for real water volume
    for i in range(length):
        if i == 0 or real_water_volume[i] != real_water_volume[i-1]:
            ax.hlines(real_water_volume[i], i, i+1, color='black', linewidth=3)
            if i != 0:
                ax.vlines(i, real_water_volume[i-1], real_water_volume[i], color='black', linewidth=3)
        elif i == length - 1:
            ax.hlines(real_water_volume[i], i, i+1, color='black', linewidth=3)
        else:
            ax.hlines(real_water_volume[i], i, i+1, color='black', linewidth=3)

    # Add patches for predicted water volume
    start = None
    end = None

    for i in range(1, len(predicted_water_volume)):
        if predicted_water_volume[i-1] == 0 and predicted_water_volume[i] != 0:
            start = i
            if i == 1 and predicted_water_volume[0] != 0:
                start = 0
        elif predicted_water_volume[i-1] != 0 and predicted_water_volume[i] == 0:
            end = i
            if start is not None:
                ax.add_patch(Rectangle((start, ax.get_ylim()[0]), end - start, predicted_water_volume[start], alpha=0.3, color='orange', fill=True))
                start = None
    ax.plot([], [], color='black', linewidth=3, label='Truth Water Volume')
    ax.add_patch(Rectangle((0, 0), 1, 1, alpha=0.3, color='orange', label='Measured Water Volume'))
    ax.legend(fontsize=16, loc='best', bbox_to_anchor=(1, 1))
    plt.show()



def plot_data_Line_with_original(data, original, title, data_range):
    data = data[data_range[0]:data_range[1]]
    original = original[data_range[0]:data_range[1]]
    lengt = len(data)
    fig, axs = plt.subplots(2)
    fig.set_size_inches(24, 8)
    fig.suptitle(title)
    x = np.linspace(0, lengt, lengt)

    plt.tick_params(labelsize=20)
    linew = 4

    a1 = data[:, 0]
    a2 = data[:, 1]
    a3 = data[:, 2]
    axs[0].plot(x, a1, label="$acc-x$", color="orange", linewidth=linew)
    axs[0].plot(x, a2, label="$acc-y$", color="blue", linewidth=linew)
    axs[0].plot(x, a3, label="$acc-z$", color="green", linewidth=linew)
    axs[0].set_ylabel('Acceleration', fontsize=20)

    g1 = data[:, 3]
    g2 = data[:, 4]
    g3 = data[:, 5]
    axs[1].plot(x, g1, label="$gyro-x$", color="black", linewidth=linew)
    axs[1].plot(x, g2, label="$gyro-y$", color="red", linewidth=linew)
    axs[1].plot(x, g3, label="$gyro-z$", color="teal", linewidth=linew)
    axs[1].set_ylabel('Angular velocity', fontsize=20)

    for ax in axs:
        ax.set_xlabel('time (s)', fontsize=20)
        ax.legend(loc='best')
        ax.grid()

        ymin, ymax = ax.get_ylim()
        ymin=0

        start = None
        for i in range(lengt):
            if original[i] == 1 and (i == 0 or original[i-1] == 0):
                start = i
                ax.vlines(i, ymin, ymax, color='black', linewidth=3)
            if original[i] == 0 and (i != 0 and original[i-1] == 1):
                ax.vlines(i, ymin, ymax, color='black', linewidth=3)
                if start is not None:
                    ax.hlines(ymax, start, i, color='black', linewidth=3)
                    start = None
            elif original[i] == 0:
                ax.hlines(ymin, i-1, i+1, color='black', linewidth=3)
    plt.show()




def segments(array):
    # Find the start and end indices of each segment
    start_indices = np.where(np.diff(array) != 0)[0] + 1
    end_indices = np.concatenate((start_indices, [len(array)]))
    
    # Combine the start and end indices into pairs, with corresponding labels
    segment_list = []
    for i in range(len(start_indices) + 1):
        segment_start = start_indices[i - 1] if i > 0 else 0
        segment_end = end_indices[i]
        segment_label = array[segment_start]
        segment_list.append((segment_start, segment_end, segment_label))
    return segment_list


    
def segment_iou(segment1, segment2):
    intersection_start = max(segment1[0], segment2[0])
    intersection_end = min(segment1[1], segment2[1])

    intersection_length = max(0, intersection_end - intersection_start)
    union_length = (segment1[1] - segment1[0]) + (segment2[1] - segment2[0]) - intersection_length

    return intersection_length / union_length

def calculate_metrics(true_segments, pred_segments, iou_threshold=0.5):
    TP, FP, FN, TN,FPnormal,FNnormal = 0, 0, 0, 0 ,0,0
    matched_pred_segments = []
    matched_true_segments = []
    FPIOU=0
    FNIOU=0
    #FPIOU：when 0riginal=0, predict=0,but IOU<desired_IOU
    #FNIOU: when Original=1,predict=1, but IOU<desired_IOU

    for t_segment in true_segments:
        t_start, t_end, t_label = t_segment
        t_length = t_end - t_start
        max_iou = 0
        max_iou_segment = None
        for p_segment in pred_segments:
            p_start, p_end, p_label = p_segment
            p_length = p_end - p_start

            if t_label != p_label:
                continue

            intersection_start = max(t_start, p_start)
            intersection_end = min(t_end, p_end)
            intersection_length = max(0, intersection_end - intersection_start)

            union_length = t_length + p_length - intersection_length

            iou = intersection_length / union_length
            if iou > max_iou:
                max_iou = iou
                max_iou_segment = p_segment

        if max_iou >= iou_threshold:
            if max_iou_segment not in matched_pred_segments:
                if t_label == 1:
                    TP += 1
                elif t_label == 0:
                    TN += 1
                matched_pred_segments.append(max_iou_segment)
                matched_true_segments.append(t_segment)
        else:
            if t_label == 1 and p_label == 1:
                FNIOU += 1
            elif t_label == 0 and p_label == 0:
                FPIOU += 1
    #FPIOU：when 0riginal=0, predict=0,but IOU<desired_IOU
    #FNIOU: when Original=1,predict=1, but IOU<desired_IOU

    
    unmatched_pred_segments = [p_segment for p_segment in pred_segments if p_segment not in matched_pred_segments]
    unmatched_true_segments = [t_segment for t_segment in true_segments if t_segment not in matched_true_segments]

    for p_segment in unmatched_pred_segments:
        p_label = p_segment[2]
        if p_label == 1:
            FPnormal += 1
        elif p_label==0:
            FNnormal+=1
    FP=FPIOU+FPnormal-FNIOU;
    FN=FNIOU+FNnormal-FPIOU

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "F1": 2 * TP / (2 * TP + FP + FN)}, matched_true_segments, matched_pred_segments


def save_metrics_to_excel(metrics_list, output_file):
    columns = ["K", "TP", "FP", "TN", "FN", "F1"]
    df = pd.DataFrame(metrics_list, columns=columns)
    df.to_excel(output_file, index=False)
    
# i=4  #跑单个文件
# path = fr"分析结果\csv_output\{i}_output.csv"
# output_file = fr"D:/迅雷下载/KU Leuven/Yiru Kong - MASTER THESIS/新喝水组/分析结果/segementResult{i}.xlsx"

# # Load your original2 and predict2 data here based on the input_path
# file = glob.glob(path)
# dl = []
# dl.append(pd.read_csv(file[0], delimiter=','))
# df = pd.concat(dl)
# data = df.drop(['H', 'y_pred'], axis=1).values
# original = df['H'].values
# predict = df['y_pred'].values
# # real_water_volume = df['RealWaterVolume'].values
# # predicted_water_volume = df['PredictWaterVolume'].values
# plotrange=(0,1000)
# originalcolor="yellow"
# predictcolor="blue"
# original2=original[30:300]
# predict2=predict[30:300]
# original_segments = segments(original2)
# predict_segments = segments(predict2)
# true_segments = segments(original)
# pred_segments = segments(predict)
# original_segments = segments(original2) #for plot
# predict_segments = segments(predict2)  #for plot
# # print(original_segments)

# metrics_list = []

# # 这里假设您已经计算出了不同iou_threshold下的metrics
# for iou_threshold in [0.25, 0.5, 0.75]:
#     metrics = calculate_metrics(true_segments, pred_segments, iou_threshold=iou_threshold)
#    #metrics_list.append([iou_threshold, metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"], metrics["F1"]])
#     metrics_dict, _, _ = calculate_metrics(original_segments, predict_segments, iou_threshold=iou_threshold)
#     metrics_list.append([iou_threshold, metrics_dict["TP"], metrics_dict["FP"], metrics_dict["TN"], metrics_dict["FN"], metrics_dict["F1"]])

# save_metrics_to_excel(metrics_list, output_file)

# metrics_dict, _, _ = calculate_metrics(true_segments, pred_segments, iou_threshold=0.25)
# print(i,"K=0.25", {key: value for key, value in metrics_dict.items() if key in ["TP", "FP", "FN", "TN", "F1"]})

# #matched_true_segments_K1={key: value for key, value in metrics_dict.items() if key in ["matched_true_segments"]}
# #matched_pred_segments_K1={key: value for key, value in metrics_dict.items() if key in ["matched_pred_segments"]}
# metrics_dict, matched_true_segments_K1, matched_pred_segments_K1 = calculate_metrics(true_segments, pred_segments, iou_threshold=0.25)
# print(i,"K=0.5", {key: value for key, value in metrics_dict.items() if key in ["TP", "FP", "FN", "TN", "F1"]})

# metrics_dict, _, _ = calculate_metrics(true_segments, pred_segments, iou_threshold=0.75)
# print(i,"K=0.75", {key: value for key, value in metrics_dict.items() if key in ["TP", "FP", "FN", "TN", "F1"]})


#plot_data(data, predict, 'Predicted Drinking Gestures', plotrange,predictcolor)
#plot_only_gestures(original, predict, 'Comparison of Original and Predicted Drinking Gestures (only gestures)', plotrange)
#plot_water_volume(real_water_volume, predicted_water_volume, 'Comparison of Real and Predicted Water Volume', plotrange)
#plot_data_Line_with_original(data, original, 'Original Drinking Gestures with Updated Line Style', plotrange)


 

  


for i in range(1, 21): #一次性跑完
    path = fr"分析结果\NEW POINT WISE EVALUATION\csv_output\{i}__output.csv"
    output_file = fr"D:/迅雷下载/KU Leuven/Yiru Kong - MASTER THESIS/新喝水组/分析结果/NEW POINT WISE EVALUATION/segementResult{i}.xlsx"

    # Load your original2 and predict2 data here based on the input_path
    file = glob.glob(path)
    dl = []
    dl.append(pd.read_csv(file[0], delimiter=','))
    df = pd.concat(dl)
    data = df.drop(['H', 'y_pred'], axis=1).values
    original = df['H'].values
    predict = df['y_pred'].values
    # real_water_volume = df['RealWaterVolume'].values
    # predicted_water_volume = df['PredictWaterVolume'].values
    plotrange=(0,1000)
    originalcolor="yellow"
    predictcolor="blue"
    original2=original[30:300]
    predict2=predict[30:300]
    original_segments = segments(original2)
    predict_segments = segments(predict2)
    true_segments = segments(original)
    pred_segments = segments(predict)
    original_segments = segments(original2) #for plot
    predict_segments = segments(predict2)  #for plot
    # print(original_segments)

    metrics_list = []

    # 这里假设您已经计算出了不同iou_threshold下的metrics
    for iou_threshold in [0.3,0.25,0.4, 0.5, 0.6, 0.75]:
        metrics = calculate_metrics(true_segments, pred_segments, iou_threshold=iou_threshold)
        #metrics_list.append([iou_threshold, metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"], metrics["F1"]])
        metrics_dict, _, _ = calculate_metrics(original_segments, predict_segments, iou_threshold=iou_threshold)
        metrics_list.append([iou_threshold, metrics_dict["TP"], metrics_dict["FP"], metrics_dict["TN"], metrics_dict["FN"], metrics_dict["F1"]])

    save_metrics_to_excel(metrics_list, output_file)
    
    metrics_dict, _, _ = calculate_metrics(true_segments, pred_segments, iou_threshold=0.25)
    print(i,"K=0.25", {key: value for key, value in metrics_dict.items() if key in ["TP", "FP", "FN", "TN", "F1"]})

    #matched_true_segments_K1={key: value for key, value in metrics_dict.items() if key in ["matched_true_segments"]}
    #matched_pred_segments_K1={key: value for key, value in metrics_dict.items() if key in ["matched_pred_segments"]}
    metrics_dict, matched_true_segments_K1, matched_pred_segments_K1 = calculate_metrics(true_segments, pred_segments, iou_threshold=0.25)
    print(i,"K=0.5", {key: value for key, value in metrics_dict.items() if key in ["TP", "FP", "FN", "TN", "F1"]})

    metrics_dict, _, _ = calculate_metrics(true_segments, pred_segments, iou_threshold=0.75)
    print(i,"K=0.75", {key: value for key, value in metrics_dict.items() if key in ["TP", "FP", "FN", "TN", "F1"]})
    count_label_1=0
    for segment in true_segments:
        if segment[2] == 1:
            count_label_1 += 1
    print("DrinkingGesture", count_label_1)
    print("DataSet",len(original))

    #plot_data(data, predict, 'Predicted Drinking Gestures', plotrange,predictcolor)
    #plot_only_gestures(original, predict, 'Comparison of Original and Predicted Drinking Gestures (only gestures)', plotrange)
    #plot_water_volume(real_water_volume, predicted_water_volume, 'Comparison of Real and Predicted Water Volume', plotrange)
    #plot_data_Line_with_original(data, original, 'Original Drinking Gestures with Updated Line Style', plotrange)
    time.sleep(2)  # 休眠 5 秒
    



# def plot_comparison(data, original, predict, title, data_range):
#     data = data[data_range[0]:data_range[1]]
#     original = original[data_range[0]:data_range[1]]
#     predict = predict[data_range[0]:data_range[1]]
#     lengt = len(data)
#     fig, axs = plt.subplots(2)
#     fig.set_size_inches(24, 8)
#     fig.suptitle(title)
#     x = np.linspace(0, lengt, lengt)

#     plt.tick_params(labelsize=20)
#     linew = 2

#     a1 = data[:, 0]
#     a2 = data[:, 1]
#     a3 = data[:, 2]
#     axs[0].plot(x, a1, label="$acc-x$", color="orange", linewidth=linew)
#     axs[0].plot(x, a2, label="$acc-y$", color="blue", linewidth=linew)
#     axs[0].plot(x, a3, label="$acc-z$", color="green", linewidth=linew)
#     axs[0].set_ylabel('Acceleration', fontsize=20)

#     g1 = data[:, 3]
#     g2 = data[:, 4]
#     g3 = data[:, 5]
#     axs[1].plot(x, g1, label="$gyro-x$", color="black", linewidth=linew)
#     axs[1].plot(x, g2, label="$gyro-y$", color="red", linewidth=linew)
#     axs[1].plot(x, g3, label="$gyro-z$", color="teal", linewidth=linew)
#     axs[1].set_ylabel('Angular velocity', fontsize=20)

#     def add_patches(ax, original_or_predict, color):
#         start = None
#         end = None

#         for i in range(1, len(original_or_predict)):
#             if original_or_predict[i-1] == 0 and original_or_predict[i] == 1:
#                 start = i
#             elif original_or_predict[i-1] == 1 and original_or_predict[i] == 0:
#                 end = i
#                 if start is not None:
#                     ax.add_patch(Rectangle((start, ax.get_ylim()[0]), end - start, ax.get_ylim()[1] - ax.get_ylim()[0], alpha=0.3, color=color))
#                     start = None

#     for ax in axs:
#         ax.set_xlabel('time (s)', fontsize=20)
#         ax.legend(loc='best')
#         ax.grid()
#         add_patches(ax, original, 'blue')
#         add_patches(ax, predict, 'yellow')

#     plt.show()

# plot_comparison(data, original, predict, 'Comparison of Original and Predicted Drinking Gestures', plotrange)