# -*- coding=utf-8 -*-
import os

def img_rename(imgs_path):
    imgs_name = os.listdir(imgs_path)
    i = 0
    for img_name in imgs_name:
        if img_name.endswith('.jpg'):
            old_name = os.path.join(os.path.abspath(imgs_path), img_name)
            # 类别+图片编号    format(str(i),'0>3s') 填充对齐
            new_name = os.path.join(os.path.abspath(imgs_path), '3' + format(str(i),'0>3s') + '.jpg')
            os.rename(old_name, new_name)
            i = i + 1
if __name__ == '__main__':
    imgs_path = 'D:/Code_Py/车牌分类/extract_plate/3/'
    img_rename(imgs_path)