"""multiple_eta
~~~~~~~~~~~~~~~

演示不同学习率对训练的影响，绘制三种 eta 下代价随训练变化的曲线。
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

# 常量
LEARNING_RATES = [0.025, 0.25, 2.5]
COLORS = ['#2A6EA6', '#FFCD33', '#FF7033']
NUM_EPOCHS = 30

def main():
    run_networks()
    make_plot()

def run_networks():
    """用三种学习率训练网络，并将代价曲线保存到 ``multiple_eta.json``。"""
    # 固定随机种子，便于复现
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    results = []
    for eta in LEARNING_RATES:
        print("\nTrain a network using eta = "+str(eta))
        net = network2.Network([784, 30, 10])
        results.append(
            net.SGD(training_data, NUM_EPOCHS, 10, eta, lmbda=5.0,
                    evaluation_data=validation_data, 
                    monitor_training_cost=True))
    f = open("multiple_eta.json", "w")
    json.dump(results, f)
    f.close()

def make_plot():
    f = open("multiple_eta.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for eta, result, color in zip(LEARNING_RATES, results, COLORS):
        _, _, training_cost, _ = result
        ax.plot(np.arange(NUM_EPOCHS), training_cost, "o-",
                label="$\eta$ = "+str(eta),
                color=color)
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
