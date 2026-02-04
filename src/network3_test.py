"""
network3_test - 简单的网络测试工具。
1、使用 mnist_loader.py 加载 MNIST 数据集，
2、使用 network3.py 训练和测试神经网络。
"""

import argparse

import os
import numpy as np

# Ensure Theano can write its compile cache and avoid C-compile issues on newer NumPy.
_flags = os.environ.get("THEANO_FLAGS", "")
_extra = "compiledir=/tmp/theano,mode=FAST_COMPILE,optimizer=fast_compile,cxx="
os.environ["THEANO_FLAGS"] = (_flags + "," + _extra).strip(",") if _flags else _extra

import mnist_loader
import numpy as np
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "sctype2char"):
    np.sctype2char = lambda obj: np.dtype(obj).char
import network3


def _shared_dataset(data):
    """将 numpy 数据封装为 theano shared 变量。"""
    x, y = data
    shared_x = network3.theano.shared(
        np.asarray(x, dtype=network3.theano.config.floatX), borrow=True)
    shared_y = network3.theano.shared(
        np.asarray(y, dtype=network3.theano.config.floatX), borrow=True)
    return shared_x, network3.T.cast(shared_y, "int32")


def build_arg_parser():
    """构建训练/测试相关的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train and test a simple neural network on MNIST.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="number of epochs to train")
    parser.add_argument("--mini-batch-size", type=int, default=10,
                        help="mini-batch size for SGD")
    parser.add_argument("--eta", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--hidden-size", type=int, default=100,
                        help="number of neurons in the hidden layer")
    parser.add_argument("--lmbda", type=float, default=0.0,
                        help="L2 regularization strength")
    # 限制样本数量，便于快速跑通与调参。
    parser.add_argument("--train-limit", type=int, default=0,
                        help="limit number of training samples (0 for all)")
    parser.add_argument("--validation-limit", type=int, default=0,
                        help="limit number of validation samples (0 for all)")
    parser.add_argument("--test-limit", type=int, default=0,
                        help="limit number of test samples (0 for all)")
    return parser


def _limit_dataset(data, limit):
    if limit and limit > 0:
        x, y = data
        return x[:limit], y[:limit]
    return data


def main():
    """加载数据，可选裁剪样本量，然后训练并评估网络。"""
    args = build_arg_parser().parse_args()
    training_data, validation_data, test_data = mnist_loader.load_data()
    training_data = _limit_dataset(training_data, args.train_limit)
    validation_data = _limit_dataset(validation_data, args.validation_limit)
    test_data = _limit_dataset(test_data, args.test_limit)

    training_data = _shared_dataset(training_data)
    validation_data = _shared_dataset(validation_data)
    test_data = _shared_dataset(test_data)

    layers = [
        network3.FullyConnectedLayer(
            784, args.hidden_size, activation_fn=network3.ReLU),
        network3.SoftmaxLayer(args.hidden_size, 10),
    ]
    net = network3.Network(layers, mini_batch_size=args.mini_batch_size)
    net.SGD(
        training_data,
        args.epochs,
        args.mini_batch_size,
        args.eta,
        validation_data,
        test_data,
        lmbda=args.lmbda,
    )


if __name__ == "__main__":
    main()
