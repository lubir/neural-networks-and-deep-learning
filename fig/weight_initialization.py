"""weight_initialization 
~~~~~~~~~~~~~~~~~~~~~~~~

展示权重初始化对训练的影响。对比两种初始化：
1) 标准差为 1 的“大权重”初始化；
2) 标准差为 1/sqrt(输入神经元数量) 的默认初始化。
"""

# 标准库
import json
import random
import sys

# 自定义库
sys.path.append('../src/')
import mnist_loader
import network2

# 第三方库
import matplotlib.pyplot as plt
import numpy as np

def main(filename, n, eta):
    run_network(filename, n, eta)
    make_plot(filename)
                       
def run_network(filename, n, eta):
    """用默认与大权重两种初始化训练网络，并将结果保存到 filename。"""
    # 固定随机种子，便于复现
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, n, 10], cost=network2.CrossEntropyCost)
    print("Train the network using the default starting weights.")
    default_vc, default_va, default_tc, default_ta \
        = net.SGD(training_data, 30, 10, eta, lmbda=5.0,
                  evaluation_data=validation_data, 
                  monitor_evaluation_accuracy=True)
    print("Train the network using the large starting weights.")
    net.large_weight_initializer()
    large_vc, large_va, large_tc, large_ta \
        = net.SGD(training_data, 30, 10, eta, lmbda=5.0,
                  evaluation_data=validation_data, 
                  monitor_evaluation_accuracy=True)
    f = open(filename, "w")
    json.dump({"default_weight_initialization":
               [default_vc, default_va, default_tc, default_ta],
               "large_weight_initialization":
               [large_vc, large_va, large_tc, large_ta]}, 
              f)
    f.close()

def make_plot(filename):
    """从 filename 读取结果并绘制图表。"""
    f = open(filename, "r")
    results = json.load(f)
    f.close()
    default_vc, default_va, default_tc, default_ta = results[
        "default_weight_initialization"]
    large_vc, large_va, large_tc, large_ta = results[
        "large_weight_initialization"]
    # 将分类正确数转换为百分比以便绘图
    default_va = [x/100.0 for x in default_va]
    large_va = [x/100.0 for x in large_va]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, 30, 1), large_va, color='#2A6EA6',
            label="Old approach to weight initialization")
    ax.plot(np.arange(0, 30, 1), default_va, color='#FFA933', 
            label="New approach to weight initialization")
    ax.set_xlim([0, 30])
    ax.set_xlabel('Epoch')
    ax.set_ylim([85, 100])
    ax.set_title('Classification accuracy')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
