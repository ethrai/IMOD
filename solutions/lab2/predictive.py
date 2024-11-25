from random import random
from math import sin
from prettytable import PrettyTable
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, W, T, A):
        self.W = W
        self.T = T
        self.A = A
        self.N = 0
        self.lastN = []

    def learn(self, input, n: int, Emin):
        # Printing
        fieldNames = ["Итерация", "Эталон", "Получено", "Отклонение"]
        table = PrettyTable(fieldNames)

        error_values = []

        iter = 0

        iterations = len(input) - n
        # TODO: refactor, crap
        self.lastN = input[-3:]
        self.N = n

        while True:
            E = 0
            for i in range(iterations):
                iter += 1
                window = input[i : i + n]
                expected = input[i + n]
                y = self.sum(window) - self.T
                cur_error = 0.5 * pow(y - expected, 2)
                E += cur_error
                for j in range(len(self.W)):
                    self.W[j] -= self.A * (y - expected) * window[j]
                self.T += self.A * (y - expected)
                table.add_row(
                    [
                        iter,
                        round(expected, 6),
                        round(y, 6),
                        format(cur_error, ".6e"),
                    ]
                )
            if E <= Emin:
                print(table)
                self.show_chart(error_values)
                return
            error_values.append(E)

    def show_chart(self, errors):
        iterations = list(range(1, len(errors) + 1))

        plt.plot(
            iterations, errors, marker="o", linestyle="-", color="b", label="Error"
        )
        plt.xlabel("Итерация")
        plt.ylabel("Среднеквадратичная ошибка")
        plt.title("Изменение среднеквадратичной ошибки по итерациям")
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, n, actual):
        fieldNames = ["Итерация", "Эталон", "Спрогнозировано", "Отклонение"]
        table = PrettyTable(fieldNames)
        iter = 0
        predict_list = [0] * (n + self.N)
        for i in range(self.N):
            predict_list[i] = self.lastN[i]
        for i in range(n):
            iter += 1
            window = predict_list[i : i + self.N]
            next = self.sum(window) - self.T
            E = 0.5 * pow(next - actual[i], 2)
            predict_list[i + self.N] = next
            table.add_row(
                [round(iter, 6), round(actual[i], 6), round(next, 6), format(E, ".6e")]
            )

        print(table)
        self.predicted_reference_chart(predict_list[self.N :], actual)
        return predict_list[self.N :]

    def predicted_reference_chart(self, predicted, reference):
        iterations = list(range(1, len(reference) + 1))

        plt.plot(
            iterations,
            reference,
            label="Эталонный ряд",
            color="blue",
            linestyle="-",
            marker="o",
        )
        plt.plot(
            iterations,
            predicted,
            label="Прогноз",
            color="red",
            linestyle="--",
            marker="x",
        )

        plt.xlabel("Итерации")
        plt.ylabel("Значения временного ряда")
        plt.title("Сравнение прогнозной кривой и эталонного ряда")
        plt.legend()
        plt.grid(True)

        plt.show()

    def sum(self, input) -> float:
        sum = 0
        for i in range(len(input)):
            sum += input[i] * self.W[i]
        return sum


def fn(a, b, d, x):
    return a * sin(b * x) + d


def tabulateFn(a, b, d, step, n):
    step = 0.1
    x = 0
    values = []
    for _ in range(n):
        values.append(fn(a, b, d, x))
        x += step
    return values


def print_lists(l1, l2):
    if len(l1) != len(l2):
        print("!!!")
    for i in range(len(l1)):
        print(l1[i], l2[i])


def print_list(l1):
    for v in enumerate(l1):
        print(v)


def main():
    A = 0.1
    Emin = 1e-12
    W = [random(), random(), random()]
    T = random()
    step = 0.1
    a = 1
    b = 5
    d = 0.1
    input = tabulateFn(a, b, d, step, 46)

    p = Perceptron(W, T, A)
    p.learn(input[:31], n=3, Emin=Emin)

    p.predict(15, input[31:])


if __name__ == "__main__":
    main()
