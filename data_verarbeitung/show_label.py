from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

#image = Image.open(r'C:\Users\13208\Desktop\Bilder_Praktikum_WS24\label\90_Grad\bilder_2710\img_1970-01-01_01-39-40.png')  

#image = Image.open(r'C:\Users\13208\Desktop\CULane\laneseg_label_w16\driver_23_30frame\05151640_0419.MP4\00510.PNG')

image = Image.open(r'C:\Users\13208\Desktop\bismarck\label\aussen\0801\img_1970-01-01_01-29-12.png') 

pixel_values = np.array(image)

plt.imshow(pixel_values, cmap='gray')

#print(pixel_values)


