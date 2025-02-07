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

def process_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    lanes = {'left': [], 'center': [], 'right': []}
    for shape in data.get('shapes', []):
        if shape['label'] == 'lane_left':
            lanes['left'] = shape['points']
        elif shape['label'] == 'lane_center':
            lanes['center'] = shape['points']
        elif shape['label'] == 'lane_right':
            lanes['right'] = shape['points']

    image = np.zeros((224, 224), dtype=np.uint8)

    for label, value in zip(['left', 'center', 'right'], [1, 2, 3]):
        if lanes[label]:
            line = Line(lanes[label])
            for y in range(224):
                x = line.getXatY(y)
                if x is not None and 0 <= x < 224:
                    image[y, x] = value

    return image

def process_directory_recursive(target_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.json'):
                # 获取JSON文件路径和对应的PNG输出路径
                json_path = os.path.join(root, file)
                json_path_less = json_path[45:]
                output_path = os.path.join(output_dir, json_path_less.replace('.json', '.png'))
                # 获取相对于目标目录的相对路径
                print(output_path)
                #print(json_path_less.replace('.json', '.png'))

                # 确保PNG文件的目录结构存在
                png_dir = os.path.dirname(output_path)
                if not os.path.exists(png_dir):
                    os.makedirs(png_dir)

                # 处理JSON文件并保存结果
                image = process_json(json_path)
                cv2.imwrite(output_path, image)

# 示例用法
input_directory = r"C:\Users\13208\Desktop\Bilder_Praktikum_WS24"  # 替换为包含JSON文件的顶级目录
output_directory = r"C:\Users\13208\Desktop\Bilder_Praktikum_WS24\label"  # 替换为保存生成PNG的顶级目录
process_directory_recursive(input_directory, output_directory)
