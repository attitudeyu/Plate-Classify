import os
import numpy as np
import tensorflow as tf
from PIL import Image
from CNN_model import inference, loss_optimizerc_accuracy

Epochs = 80
Batch_Size = 64
Resize_Width = 128
Resize_Height = 64

imgs_path = 'extract_plate/'
Summary_Dir = 'logs/'

def read_imgs_label(imgs_path):
    imgs_name = []
    labels_name = []
    # 获得图片路径和类别
    for label_name in os.listdir(imgs_path):
        for img_name in os.listdir(imgs_path+label_name):
            imgs_name.append(imgs_path+label_name+'/'+img_name)
            labels_name.append(label_name)
    # 合并图片路径和类别为二维列表 2*n
    imgs_labels = np.array([imgs_name,labels_name])
    # 转置操作 每行为图片路径和类别 n*2
    imgs_labels = imgs_labels.transpose()
    # 打乱顺序
    np.random.shuffle(imgs_labels)
    # 分别获得图片路径和类别
    random_imgs_path = list(imgs_labels[:,0])
    random_labels_name = list(imgs_labels[:,1])
    random_labels = [int(i) for i in random_labels_name]
    return random_imgs_path,random_labels

def read_imgs_to_arr(imgs_path):
    #存储图片
    imgs = []
    for img_path in imgs_path:
        img = Image.open(img_path)
        imgs.append(np.array(img))
    return imgs

# 将数据集切分为训练集和验证集
def segmentation_train_val(imgs_path,labels, ratio=0.8):
    num_imgs = len(imgs_path)
    #获得切分比例对应的索引
    idx = int(num_imgs * ratio)
    #获得训练集和验证集
    x_train = imgs_path[:idx]
    y_train = labels[:idx]
    x_val = imgs_path[idx:]
    y_val = labels[idx:]
    return x_train, y_train, x_val, y_val

# 按批次取数据
def mini_batches(imgs, labels, batch_size):
    assert len(imgs) == len(labels)
    for start_idx in range(0, len(imgs) - batch_size + 1, batch_size):
        part_idx = slice(start_idx, start_idx + batch_size)
        # 程序执行到yield语句的时候，程序暂停，
        # 返回yield后面表达式的值，在下一次调用的时候，
        # 从yield语句暂停的地方继续执行，如此循环，直到函数执行完
        yield imgs[part_idx], labels[part_idx]


if __name__ == '__main__':
    if not os.path.exists(Summary_Dir):
        os.mkdir(Summary_Dir)
    # 读取数据集(图片路径，标签列表)
    imgs_path, labels = read_imgs_label(imgs_path)

    # 切分数据集(训练集和验证集)
    x_train, y_train, x_val, y_val = segmentation_train_val(imgs_path, labels)
    x_train_num = len(x_train)
    x_val_num = len(x_val)
    print("训练集维度：",x_train_num," 验证集维度：",x_val_num)

    # 读取图像到数组中
    x_train = read_imgs_to_arr(x_train)
    x_val = read_imgs_to_arr(x_val)

    # 定义占位变量
    with tf.name_scope('input'):
        # dropout 失活率
        keep_prob = tf.placeholder(tf.float32)
        # 输入节点和标签节点
        x_node = tf.placeholder(tf.float32, shape=[None, Resize_Height, Resize_Width,3], name='x_node')
        y_node = tf.placeholder(tf.int32, shape=[None,], name='y_node')

    # 获得前向传播输出
    y_inference_out = inference(x_node,keep_prob)
    # 定义损失函数和优化算法
    train_val_loss, train_optimizer, train_val_accuracy = loss_optimizerc_accuracy(
        y_inference_out, y_node, loss_name='loss', acc_name='acc')

    # 合并所有变量操作
    merged = tf.summary.merge_all()

    # 训练神经网络
    # 建立会话
    with tf.Session() as sess:
        # 变量初始化
        sess.run(tf.global_variables_initializer())
        # 初始化写日志的writer
        train_writer = tf.summary.FileWriter(Summary_Dir+'/train', graph=sess.graph)
        # 迭代训练
        train_step = 0
        for epoch in range(Epochs):
            train_acc_sum, num_batch = 0, 0
            for x_train_batch, y_train_batch in mini_batches(x_train, y_train, Batch_Size):
                _, train_summary, train_loss_result,  train_acc_result = sess.run(
                                        [train_optimizer, merged, train_val_loss, train_val_accuracy],
                                        feed_dict={x_node: x_train_batch, y_node: y_train_batch, keep_prob: 0.5})
                train_acc_sum += train_acc_result
                num_batch += 1
                train_step +=1
                # 将训练日志写入文件
                train_writer.add_summary(train_summary, global_step=train_step)
            # 获得当前epoch每个batch样本的准确率
            print("epoch:{}".format(epoch))
            print("train accuracy:{}".format(train_acc_sum / num_batch))

            # 验证过程
            val_loss_result, val_acc_result = sess.run([train_val_loss, train_val_accuracy],
                               feed_dict={x_node: x_val, y_node: y_val, keep_prob: 1})
            print("val accuracy:{}".format(val_acc_result))
            print('*' * 50)
        train_writer.close()
