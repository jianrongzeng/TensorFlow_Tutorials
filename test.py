import tensorflow as tf
from numpy.random.mtrand import RandomState

batch_size = 8

# 声明变量,定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=2, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=2, seed=1))

# 在shape的一个维度上使用None可以方便使用不同的batch大小
x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

# 定义神经网络前向传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step = tf.train.AdadeltaOptimizer(0.1).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2) < 1] for (x1, x2) in X]

with tf.Session() as sess:
    # 初始化
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    sess.run(tf.global_variables_initializer())

    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 100000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵损失并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("after %d training step(s),current cross_entropy is %g" % (i,total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))


'''
训练神经网络的步骤分为以下三步：
1、定义神经网络的结构及输出结果；
2、定义损失函数及选择反向传播优化的算法；
3、生成会话并在训练集上返回运行反向传播优化算法
无论神经网络的结构如何变化，这三个步骤是不变的。
'''
