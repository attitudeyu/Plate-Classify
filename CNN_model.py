import tensorflow as tf

Learning_Rate = 0.001

# 卷积层
def conv_layer(input, name, kh, kw, num_out, dh, dw, set_padding='SAME'):
    # 转化输入为tensor类型
    input = tf.convert_to_tensor(input)
    # 获得输入特征图的深度
    num_in = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        # 权重矩阵的xavier初始化
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, num_in, num_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # 卷积层
        conv = tf.nn.conv2d(input, # 卷积的输入图像[batch的图片数量, 图片高度, 图片宽度, 图像通道数]
                            kernel, # 卷积核[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                            (1, dh, dw, 1), # 卷积时在图像每一维的步长
                            padding=set_padding) # 卷积方式
        # 偏差初始化
        bias_init_val = tf.constant(0.0, shape=[num_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        # 计算激活结果
        activation = tf.nn.relu(z, name=scope)
        return activation

# 全连接层
def fc_layer(input_op, name, num_out):
    num_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w',
                                 shape=[num_in, num_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[num_out], dtype=tf.float32), name='b')
        # tf.nn.relu_layer 先进行线性运算，再加上bias，最后非线性计算
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        return activation

# 池化层
def pool_labyer(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1], # 池化窗口的大小
                          strides=[1, dh, dw, 1], # 每一个维度上滑动的步长
                          padding='VALID',
                          name=name)

# 前向传播过程
def inference(input_op, keep_prob):
    # 64x128 -> 32x64x64
    conv1 = conv_layer(input_op, name="conv1", kh=11, kw=11, num_out=128, dh=1, dw=1)
    # pool1 = pool_labyer(conv1, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # 32x64x64 -> 16x32x128
    # conv2 = conv_layer(pool1, name="conv2", kh=3, kw=3, num_out=128, dh=1, dw=1)
    # pool2 = pool_labyer(conv2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # 16x32x128 -> 8x16x256 = 32768
    # conv3 = conv_layer(pool2, name="conv3", kh=3, kw=3, num_out=256, dh=1, dw=1)
    # pool3 = pool_labyer(conv3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    pool_shape = conv1.get_shape()
    flatten_shape = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(conv1, [-1, flatten_shape], name="flatten")

    # 32768 -> 128
    # fc4 = fc_layer(flatten, name="fc4", num_out=128)
    # fc4_drop = tf.nn.dropout(fc4, keep_prob, name="fc6_drop")
    # 128 -> 64
    fc5 = fc_layer(flatten, name="fc5", num_out=64)
    fc5_drop = tf.nn.dropout(fc5, keep_prob, name="fc7_drop")
    # 64 -> 4
    fc6 = fc_layer(fc5_drop, name="fc6", num_out=4)
    return fc6

# 全卷积网络
def inference_fcn(input_op, keep_prob):
    # 64x128 -> 32x64x64
    conv1 = conv_layer(input_op, name="conv1", kh=7, kw=7, num_out=64, dh=1, dw=1)
    pool1 = pool_labyer(conv1, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # 32x64x64 -> 16x32x128
    conv2 = conv_layer(pool1, name="conv2", kh=7, kw=7, num_out=128, dh=1, dw=1)
    pool2 = pool_labyer(conv2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # 16x32x128 -> 8x16x256 = 32768
    conv3 = conv_layer(pool2, name="conv3", kh=7, kw=7, num_out=256, dh=1, dw=1)
    pool3 = pool_labyer(conv3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # 8x16x256 -> 4x8x512
    conv4 = conv_layer(pool3, name="conv4", kh=7, kw=7, num_out=512, dh=1, dw=1)
    pool4 = pool_labyer(conv4, name="pool4", kh=2, kw=2, dh=2, dw=2)

    pool_shape = pool4.get_shape()
    conv_h = pool_shape[1].value
    conv_w = pool_shape[2].value
    conv5 = conv_layer(pool4, name="conv5", kh=conv_h, kw=conv_w, num_out=512, dh=1, dw=1,set_padding='VALID')

    conv6 = conv_layer(conv5, name="conv6", kh=1, kw=1, num_out=256, dh=1, dw=1,set_padding='VALID')

    conv7 = conv_layer(conv6, name="conv7", kh=1, kw=1, num_out=128, dh=1, dw=1,set_padding='VALID')

    conv8 = conv_layer(conv7, name="conv8", kh=1, kw=1, num_out=4, dh=1, dw=1,set_padding='VALID')
    flatten = tf.reshape(conv8, [-1, 4], name="flatten")
    return flatten

# 定义损失函数、优化算法、准确率
def loss_optimizerc_accuracy(cnn_out, label_out, loss_name, acc_name):
    # 标签是非稀疏表示，使用此分类交叉熵损失函数  四分类：[0,1,2,3]
    # 标签是稀疏表示，使用tf.nn.softmax_cross_entropy_with_logits损失函数  四分类：[0,0,0,1] 属于第四个分类
    train_val_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_out, logits=cnn_out))
    tf.summary.scalar(loss_name, train_val_loss)

    train_optimizer = tf.train.AdamOptimizer(learning_rate=Learning_Rate).minimize(train_val_loss)

    # 按行计算每个样本cnn_out输出的最大值索引，判断是否与标签相同
    cmp_result = tf.equal(tf.cast(tf.argmax(cnn_out, 1), tf.int32), label_out)
    # 计算均值
    train_val_accuracy = tf.reduce_mean(tf.cast(cmp_result, tf.float32))
    tf.summary.scalar(acc_name, train_val_accuracy)
    return train_val_loss, train_optimizer, train_val_accuracy

# 定义损失函数、优化算法、准确率
def loss_optimizerc_accuracy_queue(cnn_out, label_out):
    # tf.nn.softmax_cross_entropy_with_logits函数的参数labels是稀疏表示的，即[0,0,1]代表第三类
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cnn_out, labels=label_out)
    train_val_loss = tf.reduce_mean(cross_entropy)

    train_optimizer = tf.train.AdamOptimizer(Learning_Rate).minimize(train_val_loss)

    train_val_accuracy = tf.nn.in_top_k(cnn_out, # 预测输出
                              label_out, # 标签
                              1) # 取预测最大概率的索引与标签对比
    train_val_accuracy = tf.cast(train_val_accuracy, tf.float32)
    train_val_accuracy = tf.reduce_mean(train_val_accuracy)
    return train_val_loss, train_optimizer, train_val_accuracy