"""
serialize_images_to_json
~~~~~~~~~~~~~~~~~~~~~~~~

将训练/验证数据的部分样本序列化为 JSON，供 JavaScript 使用。
"""

#### 依赖库
# 标准库
import json 
import sys

# 自定义库
sys.path.append('../src/')
import mnist_loader

# 第三方库
import numpy as np


# 要序列化的训练/验证样本数量
NTD = 1000
NVD = 100

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

def make_data_integer(td):
    # 这里循环会比较慢；如果 numpy 能直接完成会更好。
    # 但 numpy.rint 后接 tolist() 仍不会转成标准 Python int。
    return [int(x) for x in (td*256).reshape(784).tolist()]

data = {"training": [
    {"x": [x[0] for x in training_data[j][0].tolist()],
     "y": [y[0] for y in training_data[j][1].tolist()]}
    for j in range(NTD)],
        "validation": [
    {"x": [x[0] for x in validation_data[j][0].tolist()],
     "y": validation_data[j][1]}
            for j in range(NVD)]}

f = open("data_1000.json", "w")
json.dump(data, f)
f.close()

