from PIL import Image
import  numpy as np
import os

def read_img_label(imgs_path):
    # 获得图片路径和类别
    for label_name in os.listdir(imgs_path):
        for img_name in os.listdir(imgs_path+label_name):
            img_path = imgs_path+label_name+'/'+img_name
            img = Image.open(img_path)
            if len(np.array(img).shape) == 2:
                print("灰度图路径：",img_path)

if __name__=='__main__':
    imgs_path = 'extract_plate/'
    read_img_label(imgs_path)