# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:04:28 2025

@author: 13208
"""
import os
import json
import random

def process_json_files(target_dir, train_list, valid_list):
    
    #if not os.path.isdir(target_dir):
        #print(f"no folder find")
        #return

    png_paths = []

    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                png_path = json_path[31:-5] + '.png'
                with open(json_path) as j:
                    data = json.load(j)

                lane_1 = 0
                lane_2 = 0
                lane_3 = 0
                lane_4 = 0
                lane_5 = 0
                lane_6 = 0

                for shape in data['shapes']:
                    if shape['label'] == 'lane_1':
                        lane_1 = 1
                    elif shape['label'] == 'lane_2':
                        lane_2 = 1
                    elif shape['label'] == 'lane_3':
                        lane_3 = 1
                    elif shape['label'] == 'lane_4':
                        lane_4 = 1
                    elif shape['label'] == 'lane_5':
                        lane_5 = 1
                    elif shape['label'] == 'lane_6':
                        lane_6 = 1

                png_path = png_path + ' \label' + png_path + ' ' + str(lane_1) + ' ' + str(lane_2) + ' ' + str(lane_3)+ ' ' + str(lane_4) + ' ' + str(lane_5) + ' ' + str(lane_6)

                png_paths.append(png_path)
    for path in png_paths:
            print(path + '\n')
    
    print(len(r"C:\Users\13208\Desktop\bismarck"))
    
    png_paths = random.shuffle(png_paths)
    
    split_index = int(len(png_paths) * 0.1)
    png_paths_valid = png_paths[:split_index]
    png_paths_train = png_paths[split_index:]

    print(len(png_paths_valid))
    print(len(png_paths_train))

    

    with open(valid_list,'w') as f:
        for path in png_paths_valid:
            f.write(path + '\n')

    with open(train_list,'w') as f:
        for path in png_paths_train:
            f.write(path + '\n')
    


if __name__ == "__main__":
    target_directory = r"C:\Users\13208\Desktop\bismarck\train+valid"
    train_txt = r"C:\Users\13208\Desktop\bismarck\train_list.txt"
    valid_txt = r"C:\Users\13208\Desktop\bismarck\valid_list.txt"
    

    process_json_files(target_directory, train_txt, valid_txt)
