"""
mnist_loader
~~~~~~~~~~~~

用于加载 MNIST 图像数据的库。返回的数据结构细节见 ``load_data``
与 ``load_data_wrapper`` 的说明。实际使用中，神经网络代码通常
调用 ``load_data_wrapper``。
"""

#### 依赖库
# 标准库
import os
import pickle
import gzip

# 第三方库
import numpy as np

def load_data():
    """返回 MNIST 数据集：(training_data, validation_data, test_data)。

    training_data 为二元组：(images, labels)。
    images 是含 50,000 个样本的 ndarray，每个样本是 784 维向量，
    对应 28*28 的像素。
    labels 是含 50,000 个标签的 ndarray，取值 0...9。

    validation_data 与 test_data 格式相同，但各只有 10,000 个样本。

    该格式便于存储，但在神经网络中使用时更适合调整格式，
    具体见 load_data_wrapper()。
    """
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mnist.pkl.gz")
    f = gzip.open(data_path, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """返回 (training_data, validation_data, test_data)。

    training_data 是 50,000 个 (x, y) 的列表：
    x 为 784 维输入向量；y 为 10 维 one-hot 向量。

    validation_data 与 test_data 是 10,000 个 (x, y) 的列表：
    x 为 784 维输入向量；y 为对应的数字标签（整数）。

    训练集与验证/测试集采用不同格式，便于在训练与评估时提高效率。
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """返回 10 维 one-hot 向量，第 j 位为 1，其余为 0。
    用于将数字 0...9 转为网络的期望输出。
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e





