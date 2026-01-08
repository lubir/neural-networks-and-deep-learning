"""network2.py
~~~~~~~~~~~~~~

network.py 的改进版，实现前馈神经网络的 SGD 训练。
改进点包括：交叉熵代价函数、正则化、更合理的权重初始化。
代码强调简洁、易读、易改；未做性能优化，也省略了不少特性。

"""

#### 依赖库
# 标准库
import json
import random
import sys

# 第三方库
import numpy as np


#### 定义二次代价与交叉熵代价

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """返回输出 a 与期望输出 y 的代价。"""
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """返回输出层的误差 delta。"""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """返回输出 a 与期望输出 y 的代价。
        使用 np.nan_to_num 保证数值稳定性。例如当 a 与 y 在同一
        位置都是 1.0 时，(1-y)*np.log(1-a) 会得到 nan，该函数会
        将其转为 0.0。
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """返回输出层的误差 delta。
        注意：参数 z 在此未使用，仅用于与其他代价类保持接口一致。
        """
        return (a-y)


#### 主网络类
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """sizes 列表给出各层神经元数量。
        例如 [2, 3, 1] 表示三层网络：输入层 2 个，隐藏层 3 个，
        输出层 1 个。偏置与权重使用 default_weight_initializer 初始化。
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """权重初始化为均值 0、标准差 1/sqrt(输入数量) 的高斯分布；
        偏置初始化为均值 0、标准差 1 的高斯分布。
        输入层不设偏置，因为偏置只用于后续层的输出计算。
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """权重使用均值 0、标准差 1 的高斯分布；偏置同样如此。
        输入层不设偏置。该初始化与第 1 章相同，仅用于对比；
        通常更推荐使用 default_weight_initializer。
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """给定输入 a，返回网络输出。"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """使用 mini-batch SGD 训练网络。
        training_data 为 (x, y) 列表，lmbda 为正则化参数。
        可传入 evaluation_data（通常是验证或测试集），并通过开关
        监控训练/评估集上的代价与准确率。返回四个列表：
        评估集代价、评估集准确率、训练集代价、训练集准确率，
        均按 epoch 统计；未开启的项返回空列表。
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            print()
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """对单个 mini-batch 做反向传播并更新权重/偏置。
        mini_batch 为 (x, y) 列表，eta 为学习率，lmbda 为正则化参数，
        n 为训练集总大小。
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """返回 (nabla_b, nabla_w)，表示代价函数 C_x 的梯度。
        nabla_b 与 nabla_w 为逐层 numpy 数组列表，结构与
        self.biases/self.weights 对齐。
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
        delta = (self.cost).delta(zs[-1], activations[-1], y)
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

    def accuracy(self, data, convert=False):
        """返回 data 中预测正确的样本数。
        输出层中激活值最大的神经元索引作为预测类别。

        convert 用于区分训练集与验证/测试集的标签表示形式。
        训练集标签为 one-hot 向量，需要转换；验证/测试集为整数标签。
        之所以不同，是出于效率考虑：训练集常用于代价计算，
        验证/测试集常用于准确率统计。详见 mnist_loader.load_data_wrapper。
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """返回数据集 data 的总代价。
        训练集时 convert=False；验证/测试集时 convert=True。
        注意与 accuracy 的 convert 约定相反，原因见上文说明。
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """将网络保存到文件 filename。"""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### 加载网络
def load(filename):
    """从文件 filename 加载网络并返回 Network 实例。"""
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### 辅助函数
def vectorized_result(j):
    """返回 10 维 one-hot 向量，第 j 位为 1，其余为 0。
    用于将数字 0-9 转为网络的期望输出。
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """Sigmoid 函数。"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Sigmoid 的导数。"""
    return sigmoid(z)*(1-sigmoid(z))
