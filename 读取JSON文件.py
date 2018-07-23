# -*- coding=utf-8 -*-
import json
import os
import numpy as np

#读取json格式文件，返回坐标
def read_json(file_name):
    file = open(file_name,'r',encoding='utf-8')
    set = json.load(file)
    # print("读取完整信息：",set)
    coord = set['objects'][0]['seg'] # 只读取第一个标注的车牌
    return coord

if __name__=='__main__':
    # 读取json文件的路径
    root = "D:/Code_Py/车牌分类/plate_json/"
    jsons_files = os.path.join(root, "green/")

    # 保存读取的真实标签路径
    labels_files = os.path.join(root, "green_label/")
    if not os.path.exists(labels_files):
        os.mkdir(labels_files)

    imgs_jsons_list = os.listdir(jsons_files)
    jsons_name = []
    # 提取图片文件夹中的json文件名称
    for idx in range(len(imgs_jsons_list)):
        if imgs_jsons_list[idx][-4:]=='json':
            jsons_name.append(imgs_jsons_list[idx])
    print("读取的json文件名称：",jsons_name)
    print("读取的json文件数量：",len(jsons_name))

    # 操作每一个json文件，读取并保存坐标
    for json_name in jsons_name:
        json_path =  os.path.join(jsons_files, json_name)
        json_coord = read_json(json_path)
        if len(json_coord)>8:
            print("标注坐标多于四个点的文件名称：",json_name)

        # 保存的坐标路径
        label_path = labels_files+json_name[:-8]+'txt'
        # 保存信息到txt文件中
        np.savetxt(label_path, json_coord)
