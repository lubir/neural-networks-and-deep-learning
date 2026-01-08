"""conv.py
~~~~~~~~~~

Michael Nielsen《Neural Networks and Deep Learning》第 6 章中卷积网络
实验相关的代码。该代码基本与书中内容一一对应，仅作为便捷实现，
因此注释较少。更多细节请参考原文。

"""

from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

import network3
from network3 import sigmoid, tanh, ReLU, Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10

def shallow(n=3, epochs=60):
    nets = []
    for j in range(n):
        print("A shallow net with 100 hidden neurons")
        net = Network([
            FullyConnectedLayer(n_in=784, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(
            training_data, epochs, mini_batch_size, 0.1, 
            validation_data, test_data)
        nets.append(net)
    return nets 

def basic_conv(n=3, epochs=60):
    for j in range(n):
        print("Conv + FC architecture")
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(
            training_data, epochs, mini_batch_size, 0.1, validation_data, test_data)
    return net 

def omit_FC():
    for j in range(3):
        print("Conv only, no FC")
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2)),
            SoftmaxLayer(n_in=20*12*12, n_out=10)], mini_batch_size)
        net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
    return net 

def dbl_conv(activation_fn=sigmoid):
    for j in range(3):
        print("Conv + Conv + FC architecture")
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=activation_fn),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=activation_fn),
            FullyConnectedLayer(
                n_in=40*4*4, n_out=100, activation_fn=activation_fn),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
    return net 

# 以下实验最终未写入章节，但保留在此作为重要的负面结果：
# 基础的 L2 正则并未显著提升效果。原因（我认为）是
# 卷积+池化本身已经是很强的正则化。
def regularized_dbl_conv():
    for lmbda in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for j in range(3):
            print("Conv + Conv + FC num %s, with regularization %s" % (j, lmbda))
            net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                              filter_shape=(20, 1, 5, 5), 
                              poolsize=(2, 2)),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                              filter_shape=(40, 20, 5, 5), 
                              poolsize=(2, 2)),
                FullyConnectedLayer(n_in=40*4*4, n_out=100),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
            net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data, lmbda=lmbda)

def dbl_conv_relu():
    for lmbda in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for j in range(3):
            print("Conv + Conv + FC num %s, relu, with regularization %s" % (j, lmbda))
            net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                              filter_shape=(20, 1, 5, 5), 
                              poolsize=(2, 2), 
                              activation_fn=ReLU),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                              filter_shape=(40, 20, 5, 5), 
                              poolsize=(2, 2), 
                              activation_fn=ReLU),
                FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
            net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=lmbda)

#### 后续函数可能会使用扩展后的 MNIST 数据。
#### 可通过运行 expand_mnist.py 生成。

def expanded_data(n=100):
    """n 为全连接层神经元数量，这里会尝试 100、300、1000。"""
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
    for j in range(3):
        print("Training with expanded data, %s neurons in the FC layer, run num %s" % (n, j))
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            FullyConnectedLayer(n_in=40*4*4, n_out=n, activation_fn=ReLU),
            SoftmaxLayer(n_in=n, n_out=10)], mini_batch_size)
        net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, 
                validation_data, test_data, lmbda=0.1)
    return net 

def expanded_data_double_fc(n=100):
    """n 为两个全连接层的神经元数量，这里会尝试 100、300、1000。"""
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
    for j in range(3):
        print("Training with expanded data, %s neurons in two FC layers, run num %s" % (n, j))
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            FullyConnectedLayer(n_in=40*4*4, n_out=n, activation_fn=ReLU),
            FullyConnectedLayer(n_in=n, n_out=n, activation_fn=ReLU),
            SoftmaxLayer(n_in=n, n_out=10)], mini_batch_size)
        net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, 
                validation_data, test_data, lmbda=0.1)

def double_fc_dropout(p0, p1, p2, repetitions):
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
    nets = []
    for j in range(repetitions):
        print("\n\nTraining using a dropout network with parameters ", p0, p1, p2)
        print("Training with expanded data, run num %s" % j)
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2), 
                          activation_fn=ReLU),
            FullyConnectedLayer(
                n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=p0),
            FullyConnectedLayer(
                n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=p1),
            SoftmaxLayer(n_in=1000, n_out=10, p_dropout=p2)], mini_batch_size)
        net.SGD(expanded_training_data, 40, mini_batch_size, 0.03, 
                validation_data, test_data)
        nets.append(net)
    return nets

def ensemble(nets): 
    """输入多个网络并进行投票集成。
    返回一个元组：测试集中预测错误的样本索引，以及对应的错误预测。
    这里使用了快速但不够优雅的实现；更合理的做法是用 Theano
    定义一个投票函数，但当前实现足够使用。
    """
    
    test_x, test_y = test_data
    for net in nets:
        i = T.lscalar() # 小批量索引
        net.test_mb_predictions = theano.function(
            [i], net.layers[-1].y_out,
            givens={
                net.x: 
                test_x[i*net.mini_batch_size: (i+1)*net.mini_batch_size]
            })
        net.test_predictions = list(np.concatenate(
            [net.test_mb_predictions(i) for i in range(1000)]))
    all_test_predictions = zip(*[net.test_predictions for net in nets])
    def plurality(p): return Counter(p).most_common(1)[0][0]
    plurality_test_predictions = [plurality(p) 
                                  for p in all_test_predictions]
    test_y_eval = test_y.eval()
    error_locations = [j for j in range(10000) 
                       if plurality_test_predictions[j] != test_y_eval[j]]
    erroneous_predictions = [plurality(all_test_predictions[j])
                             for j in error_locations]
    print("Accuracy is {:.2%}".format((1-len(error_locations)/10000.0)))
    return error_locations, erroneous_predictions

def plot_errors(error_locations, erroneous_predictions=None):
    test_x, test_y = test_data[0].eval(), test_data[1].eval()
    fig = plt.figure()
    error_images = [np.array(test_x[i]).reshape(28, -1) for i in error_locations]
    n = min(40, len(error_locations))
    for j in range(n):
        ax = plt.subplot2grid((5, 8), (j/8, j % 8))
        ax.matshow(error_images[j], cmap = matplotlib.cm.binary)
        ax.text(24, 5, test_y[error_locations[j]])
        if erroneous_predictions:
            ax.text(24, 24, erroneous_predictions[j])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt
    
def plot_filters(net, layer, x, y):

    """绘制指定卷积层后的滤波器。
    以 x*y 的网格形式展示。例如 layer 0 有 20 个滤波器时，
    可调用 show_filters(net, 0, 5, 4) 来绘制 5x4 的滤波器图。
    """
    filters = net.layers[layer].w.eval()
    fig = plt.figure()
    for j in range(len(filters)):
        ax = fig.add_subplot(y, x, j)
        ax.matshow(filters[j][0], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt


#### 运行书中所有实验的辅助方法

def run_experiments():

    """运行书中描述的实验。
    注意后续实验需要扩展训练数据，可通过 expand_mnist.py 生成。
    """
    shallow()
    basic_conv()
    omit_FC()
    dbl_conv(activation_fn=sigmoid)
    # 章节中省略但仍有参考价值的实验：regularized_dbl_conv()
    dbl_conv_relu()
    expanded_data(n=100)
    expanded_data(n=300)
    expanded_data(n=1000)
    expanded_data_double_fc(n=100)    
    expanded_data_double_fc(n=300)
    expanded_data_double_fc(n=1000)
    nets = double_fc_dropout(0.5, 0.5, 0.5, 5)
    # 绘制刚训练完成的集成模型中误分类的数字
    error_locations, erroneous_predictions = ensemble(nets)
    plt = plot_errors(error_locations, erroneous_predictions)
    plt.savefig("ensemble_errors.png")
    # 绘制刚训练完成的第一个网络学到的滤波器
    plt = plot_filters(nets[0], 0, 5, 4)
    plt.savefig("net_full_layer_0.png")
    plt = plot_filters(nets[0], 1, 8, 5)
    plt.savefig("net_full_layer_1.png")
