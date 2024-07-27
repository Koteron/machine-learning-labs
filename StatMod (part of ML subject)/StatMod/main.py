import random
from prettytable import PrettyTable
from matplotlib import pyplot as plt


def evaluate_correlation_coef(rand_num_array, expectation, f):
    sum1 = 0.0
    for i in range(0, len(rand_num_array) - f):
        sum1 += (rand_num_array[i] - expectation) * (rand_num_array[i + f] - expectation)
    sum2 = 0.0
    for number in rand_num_array:
        sum2 += (number - expectation) ** 2
    return sum1 / sum2


def evaluate_metrics(amount, table):
    sum1 = 0.0
    rand_number_array = list()
    for i in range(amount):
        rand_number_array.append(random.uniform(0, 1))
        sum1 += rand_number_array[i]

    expectation = sum1 / amount

    dispersion = 0.0
    for number in rand_number_array:
        dispersion += (number - expectation) ** 2

    dispersion /= amount

    table.add_row([amount, expectation, 0.5, 0.5 - expectation,
                   dispersion, 1 / 12, 1 / 12 - dispersion])

    correlation_array = list()
    for i in range(1, amount):
        correlation_array.append(evaluate_correlation_coef(rand_number_array, expectation, i))
    plt.bar(range(1, amount), correlation_array)
    plt.show()
    rand_number_array.sort()
    counter_arr = list()
    for number in rand_number_array:
        counter = 0
        for number2 in rand_number_array:
            if number >= number2:
                counter += 1
        counter_arr.append(counter/amount)
    plt.bar(range(1, amount + 1), counter_arr)
    plt.show()
    count_array = list()
    last_i = 0
    for j in range(1, 11):
        counter = 0
        for i in range(last_i, len(rand_number_array)):
            if rand_number_array[i] < float(j) / 10:
                counter += 1
                last_i = i
        count_array.append(counter / amount)
    plt.bar([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], count_array, 0.1)
    plt.show()


table = PrettyTable()
table.field_names = ["Amount", "Experiment M", "Theoretic M", "Deviation M",
                     "Experiment D", "Theoretic D", "Deviation D"]
evaluate_metrics(10, table)
evaluate_metrics(100, table)
evaluate_metrics(1000, table)
evaluate_metrics(10000, table)

print(table)
