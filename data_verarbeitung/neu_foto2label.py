# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:08:40 2025

@author: 13208
"""

import os
import json
import numpy as np
import cv2

class Line:
    def __init__(self, points):
        self.points = points

    def getXatY(self, y):
        if len(self.points) < 2:
            return None
        self.points.sort(key=lambda p: p[1], reverse=True)

        if y >= self.points[0][1]:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
        elif y <= self.points[-1][1]:
            x1, y1 = self.points[-2]
            x2, y2 = self.points[-1]
        else:
            for i in range(1, len(self.points)):
                if y > self.points[i][1]:
                    x1, y1 = self.points[i - 1]
                    x2, y2 = self.points[i]
                    break
        if y1 == y2:
            return None
        m = (x2 - x1) / (y2 - y1)
        b = x1 - m * y1
        return int(m * y + b)

def process_json(file_path, thickness=1):
    with open(file_path, 'r') as f:
        data = json.load(f)

    lanes = {'lane_1': [], 'lane_2': [], 'lane_3': [], 'lane_4': [], 'lane_5': [], 'lane_6': []}
    for shape in data.get('shapes', []):
        if shape['label'] == 'lane_1':
            lanes['lane_1'] = shape['points']
        elif shape['label'] == 'lane_2':
            lanes['lane_2'] = shape['points']
        elif shape['label'] == 'lane_3':
            lanes['lane_3'] = shape['points']
        elif shape['label'] == 'lane_4':
            lanes['lane_4'] = shape['points']
        elif shape['label'] == 'lane_5':
            lanes['lane_5'] = shape['points']
        elif shape['label'] == 'lane_6':
            lanes['lane_6'] = shape['points']

    image = np.zeros((224, 224), dtype=np.uint8)

    for label, value in zip(['lane_1', 'lane_2', 'lane_3', 'lane_4', 'lane_5', 'lane_6'], [1, 2, 3, 4, 5, 6]):
        if lanes[label]:
            line = Line(lanes[label])
            # 获取当前线段的最小和最大 y 值
            min_y = min(point[1] for point in lanes[label])
            max_y = max(point[1] for point in lanes[label])
            for y in range(224):
                if y < min_y - 10 or y > max_y + 10:
                    continue
                x = line.getXatY(y)
                if x is not None and 0 <= x < 224:
                    for t in range(-thickness, thickness + 1):
                        if 0 <= x + t < 224:
                            image[y, x + t] = value

    return image

def process_directory_recursive(target_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.json'):
                # 获取JSON文件路径和对应的PNG输出路径
                json_path = os.path.join(root, file)
                json_path_less = json_path[32:]#数字根据路径调整
                output_path = os.path.join(output_dir, json_path_less.replace('.json', '.png'))
                # 获取相对于目标目录的相对路径
                print(output_path)
                #print(json_path_less.replace('.json', '.png'))

                # 确保PNG文件的目录结构存在
                png_dir = os.path.dirname(output_path)
                if not os.path.exists(png_dir):
                    os.makedirs(png_dir)

                # 处理JSON文件并保存结果
                image = process_json(json_path, 1)
                cv2.imwrite(output_path, image)

# 示例用法
input_directory = r"C:\Users\13208\Desktop\bismarck"  # 替换为包含JSON文件的顶级目录
output_directory = r"C:\Users\13208\Desktop\bismarck\label"  # 替换为保存生成PNG的顶级目录
process_directory_recursive(input_directory, output_directory)
