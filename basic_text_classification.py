from tensorflow.contrib import keras
import tensorflow as tf

imdb = keras.datasets.imdb
# 参数 num_words=10000 会保留训练数据中出现频次在前 10000 位的字词。为确保数据规模处于可管理的水平，罕见字词将被舍弃。
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 该数据集已经过预处理：每个样本都是一个整数数组，表示影评中的字词。每个标签都是整数值 0 或 1，其中 0 表示负面影评，1 表示正面影评。
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# 影评文本已转换为整数，其中每个整数都表示字典中的一个特定字词。第一条影评如下所示：
print(train_data[0])

# 创建一个辅助函数来查询包含整数到字符串映射的字典对象：
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# The first indices are reserved
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))

# 影评（整数数组）必须转换为张量，然后才能馈送到神经网络中。我们可以通过以下两种方法实现这种转换：
# 对数组进行独热编码，将它们转换为由 0 和 1 构成的向量。例如，序列 [3, 5] 将变成一个 10000 维的向量，除索引3和5
# 转换为1之外，其余全转换为 0。然后，将它作为网络的第一层，一个可以处理浮点向量数据的密集层。不过，这种方法会占用
# 大量内存，需要一个大小为 num_words * num_reviews 的矩阵。
# 或者，我们可以填充数组，使它们都具有相同的长度，然后创建一个形状为 max_length * num_reviews 的整数张量。
# 我们可以使用一个能够处理这种形状的嵌入层作为网络中的第一层。
# 在本教程中，我们将使用第二种方法。
# 由于影评的长度必须相同，我们将使用 pad_sequences 函数将长度标准化：
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


# 样本的长度：
# len(train_data[0]), len(train_data[1])
# 检查（现已填充的）第一条影评：
# print(train_data[0])


# 构建模型
# 神经网络通过堆叠层创建而成，这需要做出两个架构方面的主要决策：
# 要在模型中使用多少个层？
# 要针对每个层使用多少个隐藏单元？
# 在本示例中，输入数据由字词-索引数组构成。要预测的标签是 0 或 1。接下来，我们为此问题构建一个模型：

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

# 配置模型以使用优化器和损失函数：
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 创建验证集
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 评估模型
results = model.evaluate(test_data, test_labels)
print(results)

# 创建准确率和损失随时间变化的图
history_dict = history.history
history_dict.keys()


