import tensorflow as tf

# 定义一个LSTM结构域
lstm = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)

# 将LSTM中的状态初始化为全0数组
# state是一个包含两个张量的LSTMStateTuple类，其中state.c、state.h分别代表c状态和h状态
# batch_size给出了一个batch训练样本的大小
state = lstm.zero_state(batch_size, tf.float32)

# 定义损失函数
loss = 0.0
# 用num_steps来表示训练数据的序列长度
for i in range(num_steps):
    # 在第一个时刻声明LSTM中使用到的变量，在之后的时刻都需要复用之前定义好的变量
    if i > 0:
        tf.get_variable_scope().reuse_variables()

    lstm_output, state = lstm(current_input, state)
    final_output = fully_connected(lstm_output)
    loss += calc_loss(final_output, expected_output)

# 使用类似第四章中介绍的方法训练模型