import numpy as np
import os
from PIL import Image

Resize_Width = 128
Resize_Height = 64

def read_label(label_path):
    labels = []
    # 读取标签
    with open(label_path, 'r', encoding='UTF-8') as label:
        # 读取当前txt文件的所有内容
        label_lines = label.readlines()
    label = []
    # 将当前txt文件的每行切割
    for idx, line in enumerate(label_lines):
        one_line = line.strip().split('\n')
        one_line = float(one_line[0])
        label.extend([one_line])
    labels.append(label)
    return np.array(labels,np.float32).reshape((4,2))

# 保存裁剪后的图像区域和坐标
def save_resize_img(extract_img_path,extract_img):
    # PIL保存数组为图片需要先进行格式转化
    extract_img = Image.fromarray(extract_img)
    extract_img = extract_img.resize((Resize_Width, Resize_Height), Image.ANTIALIAS)
    extract_img.save(extract_img_path)

# 裁剪图像
def extract_image(img, img_box):
    # 裁剪目标区域
    x_arr = [int(i[0]) for i in img_box]
    y_arr = [int(i[1]) for i in img_box]
    x_min = min(x_arr)
    x_max = max(x_arr)
    y_min = min(y_arr)
    y_max = max(y_arr)
    extract_img = img[y_min:y_max, x_min:x_max]
    #print("裁剪图像的维度：",extract_img.shape)
    return extract_img

if __name__=='__main__':
    root = "D:/Code_Py/车牌分类/plate_json/"
    imgs_path = os.path.join(root, "green_img/")
    labels_path = os.path.join(root, "green_label/")

    # 提取车牌区域后的保存路径
    extract_root = "D:/Code_Py/车牌分类/extract_plate/"
    extract_imgs_path = os.path.join(extract_root,"green/")
    if not os.path.exists(extract_imgs_path):
        os.mkdir(extract_imgs_path)

    imgs_name = os.listdir(imgs_path)
    extract_img_h_sum = 0
    extract_img_w_sum = 0
    img_num = len(imgs_name)

    # 操作每一张图片
    for img_name in imgs_name:
        img_path = os.path.join(imgs_path,img_name)
        extract_img_path = os.path.join(extract_imgs_path, img_name)
        label_name = img_name[:-4]+'.txt'
        extract_label_path = os.path.join(labels_path, label_name)

        # 读取图片
        img = np.array(Image.open(img_path))
        # 获得车牌标注坐标
        img_box = read_label(extract_label_path)
        # 裁剪获得新图像
        extract_img = extract_image(img, img_box)
        # 统计车牌图像的宽和高
        extract_img_h_sum += extract_img.shape[0]
        extract_img_w_sum += extract_img.shape[1]
        # 对图像进行resize到固定大小
        save_resize_img(extract_img_path, extract_img)

    print("计算车牌的高度均值和宽度均值：",
          extract_img_h_sum/img_num,extract_img_w_sum/img_num)

