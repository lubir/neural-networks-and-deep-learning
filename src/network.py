"""
network.py
~~~~~~~~~~

用于实现前馈神经网络的随机梯度下降（SGD）学习算法。
梯度通过反向传播计算。这里的实现重点是简洁、易读、易改。
代码未做性能优化，也省略了很多实际工程中常用的特性。
"""

#### 依赖库
# 标准库
import random

# 第三方库
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """sizes 列表给出各层神经元数量。
        例如 [2, 3, 1] 表示三层网络：输入层 2 个神经元，隐藏层 3 个，
        输出层 1 个。偏置与权重用均值 0、方差 1 的高斯分布随机初始化。
        约定输入层不使用偏置，因为偏置只用于后续层的输出计算。
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """给定输入 a，返回网络输出。"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """使用 mini-batch SGD 训练网络。
        training_data 为 (x, y) 组成的列表，分别是输入与目标输出。
        其他参数含义直接见名称。如果提供 test_data，将在每个 epoch
        结束后在测试集上评估并打印进度（会显著降低速度）。
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """对单个 mini-batch 做反向传播并更新权重/偏置。
        mini_batch 是 (x, y) 列表，eta 为学习率。
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """返回 (nabla_b, nabla_w)，表示代价函数 C_x 的梯度。
        nabla_b 与 nabla_w 为逐层的 numpy 数组列表，结构与
        self.biases 和 self.weights 对齐。
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前向传播
        activation = x
        activations = [x] # 按层存储激活值
        zs = [] # 按层存储 z 向量
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 反向传播
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 注意：这里的 l 与书中第 2 章的记号略不同。
        # 这里 l=1 表示最后一层，l=2 表示倒数第二层，依此类推。
        # 这样编号可以利用 Python 列表的负索引特性。
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """返回测试集中预测正确的样本数量。
        假设输出层中激活值最大的神经元索引即为预测类别。
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """返回输出层激活的偏导向量：∂C_x/∂a。"""
        return (output_activations-y)

#### 辅助函数
def sigmoid(z):
    """Sigmoid 函数。"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Sigmoid 的导数。"""
    return sigmoid(z)*(1-sigmoid(z))
