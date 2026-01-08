"""
mnist_average_darkness
~~~~~~~~~~~~~~~~~~~~~~

一个非常朴素的 MNIST 手写数字分类器：按“图像平均暗度”来分类。
直觉是“1”通常比“8”更亮，因为后者形状更复杂、笔画更多。
给定一张图像，分类器返回训练集中平均暗度最接近的数字。

程序分两步：先在训练集上计算每个数字的平均暗度，
再在测试集上评估分类准确率。

显然这不是一个好的分类方法，但它能展示朴素方法的效果基线。
"""

#### 依赖库
# 标准库
from collections import defaultdict

# 自定义库
import mnist_loader

def main():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # 训练阶段：计算每个数字的平均暗度
    avgs = avg_darknesses(training_data)
    # 测试阶段：统计测试集的正确分类数量
    num_correct = sum(int(guess_digit(image, avgs) == digit)
                      for image, digit in zip(test_data[0], test_data[1]))
    print("Baseline classifier using average darkness of image.")
    print("%s of %s values correct." % (num_correct, len(test_data[1])))

def avg_darknesses(training_data):
    """返回一个 defaultdict，键为 0~9。
    对每个数字计算训练集中该数字图像的平均暗度。
    单张图像的暗度即像素值之和。
    """
    digit_counts = defaultdict(int)
    darknesses = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darknesses[digit] += sum(image)
    avgs = defaultdict(float)
    for digit, n in digit_counts.items():
        avgs[digit] = darknesses[digit] / n
    return avgs

def guess_digit(image, avgs):
    """返回训练集平均暗度最接近 image 的数字。
    avgs 为 defaultdict，键为 0...9，值为对应数字的平均暗度。
    """
    darkness = sum(image)
    distances = {k: abs(v-darkness) for k, v in avgs.items()}
    return min(distances, key=distances.get)

if __name__ == "__main__":
    main()
