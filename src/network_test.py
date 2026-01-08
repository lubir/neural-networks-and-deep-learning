"""
network_test - 简单的网络测试工具。
1、使用 mnist_loader.py 加载 MNIST 数据集，
2、使用 network.py 训练和测试神经网络。
"""

import argparse

import mnist_loader
import network


def build_arg_parser():
    """构建训练/测试相关的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train and test a simple neural network on MNIST.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="number of epochs to train")
    parser.add_argument("--mini-batch-size", type=int, default=10,
                        help="mini-batch size for SGD")
    parser.add_argument("--eta", type=float, default=3.0,
                        help="learning rate")
    parser.add_argument("--hidden-size", type=int, default=30,
                        help="number of neurons in the hidden layer")
    # 限制样本数量，便于快速跑通与调参。
    parser.add_argument("--train-limit", type=int, default=0,
                        help="limit number of training samples (0 for all)")
    parser.add_argument("--test-limit", type=int, default=0,
                        help="limit number of test samples (0 for all)")
    return parser


def main():
    """加载数据，可选裁剪样本量，然后训练并评估网络。"""
    args = build_arg_parser().parse_args()
    training_data, _, test_data = mnist_loader.load_data_wrapper()
    # 可选缩小数据集，加速本地测试。
    if args.train_limit > 0:
        training_data = training_data[:args.train_limit]
    if args.test_limit > 0:
        test_data = test_data[:args.test_limit]
    net = network.Network([784, args.hidden_size, 10])
    net.SGD(training_data, args.epochs, args.mini_batch_size, args.eta,
            test_data=test_data)


if __name__ == "__main__":
    main()
