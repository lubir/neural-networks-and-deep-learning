"""expand_mnist.py
~~~~~~~~~~~~~~~~~~

将 50,000 张 MNIST 训练图片扩展为 250,000 张：对每张图上下左右
各平移 1 个像素。结果保存到 ../data/mnist_expanded.pkl.gz。

注意：该程序占用内存较多，小型机器可能无法运行。
"""

from __future__ import print_function

#### 依赖库

# 标准库
import pickle
import gzip
import os.path
import random

# 第三方库
import numpy as np

print("正在扩展 MNIST 训练集")

if os.path.exists("../data/mnist_expanded.pkl.gz"):
    print("扩展后的训练集已存在，退出。")
else:
    f = gzip.open("../data/mnist.pkl.gz", 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    expanded_training_pairs = []
    j = 0 # 计数器
    for x, y in zip(training_data[0], training_data[1]):
        expanded_training_pairs.append((x, y))
        image = np.reshape(x, (-1, 28))
        j += 1
        if j % 1000 == 0: print("已扩展图像数量", j)
        # 根据位移参数进行像素平移
        for d, axis, index_position, index in [
                (1,  0, "first", 0),
                (-1, 0, "first", 27),
                (1,  1, "last",  0),
                (-1, 1, "last",  27)]:
            new_img = np.roll(image, d, axis)
            if index_position == "first": 
                new_img[index, :] = np.zeros(28)
            else: 
                new_img[:, index] = np.zeros(28)
            expanded_training_pairs.append((np.reshape(new_img, 784), y))
    random.shuffle(expanded_training_pairs)
    expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
    print("正在保存扩展数据，可能需要几分钟。")
    f = gzip.open("../data/mnist_expanded.pkl.gz", "wb")
    pickle.dump((expanded_training_data, validation_data, test_data), f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
