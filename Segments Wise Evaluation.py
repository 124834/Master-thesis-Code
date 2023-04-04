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

path = r'分离数据1\output.csv' #change to the path you need
file = glob.glob(path)
dl = []
dl.append(pd.read_csv(file[0], delimiter=','))
df = pd.concat(dl)
data = df.drop(['original', 'predict'], axis=1).values
original = df['original'].values
predict = df['predict'].values
plotrange=(30,300)
originalcolor="yellow"
predictcolor="blue"
original2=original[30:300]
predict2=predict[30:300]

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

plot_data(data, predict, 'Predicted Drinking Gestures', plotrange,predictcolor)




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

    add_patches(original, 'black', fill=False)
    add_patches(predict, 'blue', fill=True)

    
    for i in range(lengt):
        if original[i] == 1 and original[i-1]==0 :
            ax.vlines(i, 0, 1, color='black', linewidth=3)
        if original[i] == 0 and original[i-1]==1 :
            ax.vlines(i, 0, 1, color='black', linewidth=3)

    plt.show()

plot_only_gestures(original, predict, 'Comparison of Original and Predicted Drinking Gestures (only gestures)', plotrange)

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

plot_data_Line_with_original(data, original, 'Original Drinking Gestures with Updated Line Style', plotrange)

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

original_segments = segments(original2)
predict_segments = segments(predict2)
# print(original_segments)
    
def segment_iou(segment1, segment2):
    intersection_start = max(segment1[0], segment2[0])
    intersection_end = min(segment1[1], segment2[1])

    intersection_length = max(0, intersection_end - intersection_start)
    union_length = (segment1[1] - segment1[0]) + (segment2[1] - segment2[0]) - intersection_length

    return intersection_length / union_length

def calculate_metrics(true_segments, pred_segments, iou_threshold=0.5):
    TP, FP, FN, TN = 0, 0, 0, 0
    matched_pred_segments = []
    
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
            TP += 1
            matched_pred_segments.append(max_iou_segment)
        else:
            if t_label == 1:
                FN += 1
            else:
                FP += 1
    
    for p_segment in pred_segments:
        if p_segment not in matched_pred_segments:
            p_label = p_segment[2]
            if p_label == 1:
                FP += 1
            else:
                TN += 1
    
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN, "F1": 2 * TP / (2 * TP + FP + FN)}


true_segments = segments(original)
pred_segments = segments(predict)

metrics = calculate_metrics(true_segments, pred_segments, iou_threshold=0.25)
print("K=0.25" , metrics)

metrics = calculate_metrics(true_segments, pred_segments, iou_threshold=0.5)
print("K=0.5" , metrics)

metrics = calculate_metrics(true_segments, pred_segments, iou_threshold=0.75)
print("K=0.75" , metrics)

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