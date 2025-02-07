import os
import json

def process_json_files(target_dir, train_list, valid_list):
    
    #if not os.path.isdir(target_dir):
        #print(f"no folder find")
        #return

    png_paths = []

    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                png_path = json_path[44:-5] + '.png'
                with open(json_path) as j:
                    data = json.load(j)

                has_left = 0
                has_center = 0
                has_right = 0

                for shape in data['shapes']:
                    if shape['label'] == 'lane_left':
                        has_left = 1
                    elif shape['label'] == 'lane_center':
                        has_center = 1
                    elif shape['label'] == 'lane_right':
                        has_right = 1

                png_path = png_path + ' \label' + png_path + ' ' + str(has_left) + ' ' + str(has_center) + ' ' + str(has_right)

                png_paths.append(png_path)
    for path in png_paths:
            print(path + '\n')
    
    print(len(r"C:\Users\13208\Desktop\Bilder_Praktikum_WS24"))
    
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
    target_directory = r"C:\Users\13208\Desktop\Bilder_Praktikum_WS24"
    train_txt = r"C:\Users\13208\Desktop\Bilder_Praktikum_WS24\train_list.txt"
    valid_txt = r"C:\Users\13208\Desktop\Bilder_Praktikum_WS24\valid_list.txt"
    

    process_json_files(target_directory, train_txt, valid_txt)