from random import random

import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    def __init__(self, W, T, A):
        self.W = W
        self.T = T
        self.A = A

    def learn(self, inputs, expected):
        output = [0] * len(inputs)
        count = 0
        while output != expected:
            for i, input in enumerate(inputs):
                count += 1
                y = self.run(input)
                output[i] = y
                for j in range(len(input)):
                    self.W[j] -= self.A * input[j] * (y - expected[i])
                self.T += self.A * (y - expected[i])
        print(count)

    def run(self, input):
        return self.produce(self.sum(input))

    def produce(self, sum):
        return 1 if sum >= 0 else 0

    def sum(self, input):
        sum = 0
        for i in range(len(input)):
            sum += input[i] * self.W[i]
        return sum - self.T


def main():
    A = 0.1
    W = [random(), random()]
    T = random()
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected = [0, 1, 1, 1]

    p = Perceptron(W, T, A)
    p.learn(inputs, expected)

    W = np.array([p.W[0], p.W[1]])  # Веса
    T = p.T  # Порог

    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 1])  # Ожидаемые выходы для операции ИЛИ

    def decision_boundary(x):
        # Выражаем x2 через x1 из уравнения w1*x1 + w2*x2 = T
        return (T - W[0] * x) / W[1]

    plt.figure(figsize=(6, 6))

    for i, point in enumerate(data):
        if labels[i] == 1:
            plt.plot(point[0], point[1], "bo")  # Синие точки для метки 1 (выход 1)
        else:
            plt.plot(point[0], point[1], "ro")  # Красные точки для метки 0 (выход 0)
    x_vals = np.linspace(-0.5, 1.5, 100)
    y_vals = decision_boundary(x_vals)

    plt.plot(x_vals, y_vals, "k-", label="Граница раздела")

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.title("Решение однослойного перцептрона для логического ИЛИ")
    plt.savefig("line_plot.png")


if __name__ == "__main__":
    main()
