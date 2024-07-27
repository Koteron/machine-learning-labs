import random
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import math


def uniform_distribution(min_val, max_val):
    return int(round(max_val - min_val + 1) * random.uniform(0, 1) + min_val)


def uniform_distribution_array(min_val, max_val, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(uniform_distribution(min_val, max_val))
    return rand_number_arr


def binomial_distribution(n, p):
    if n >= 100:
        sum = 0
        for i in range(0, 28):
            sum += random.uniform(0, 1)
        return round(2 ** 0.5 * ((n * p * (1 - p)) ** 0.5 + 0.5) * (sum - 14) / 2.11233 + n * p)
    a1 = random.uniform(0, 1)
    p_r = (1 - p) ** n
    r = 0
    while a1 - p_r >= 0:
        a1 = a1 - p_r
        p_r *= (p * (n - r)) / ((r + 1) * (1 - p))
        r += 1
    return r


def binomial_distribution_array(n, p, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(binomial_distribution(n, p))
    return rand_number_arr


def geometric_1(p):
    a = random.uniform(0, 1)
    p_t = p
    r = 1
    while a - p_t >= 0:
        a -= p_t
        p_t *= (1 - p)
        r += 1
    return r


def geometric_2(p):
    a = random.uniform(0, 1)
    r = 1
    while a > p:
        a = random.uniform(0, 1)
        r += 1
    return r


def geometric_3(p):
    a = random.uniform(0, 1)
    return int(math.log(a) / math.log(1 - p)) + 1


def normal_distribution(m, sko):
    sum = 0
    for i in range(0, 28):
        sum += random.uniform(0, 1)
    return math.sqrt(2) * sko * (sum - 14) / 2.11233 + m


def poisson_1(mu):
    if mu >= 88:
        return normal_distribution(mu, mu)
    p_t = math.exp(-mu)
    a = random.uniform(0, 1)
    r = 1
    while a - p_t >= 0:
        a -= p_t
        p_t *= mu / r
        r += 1
    return r


def poisson_2(mu):
    if mu >= 88:
        return normal_distribution(mu, mu)
    a = random.uniform(0, 1)
    p_t = a
    r = 1
    while p_t >= math.exp(-mu):
        a = random.uniform(0, 1)
        p_t *= a
        r += 1
    return r


def poisson_1_array(mu, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(poisson_1(mu))
    return rand_number_arr


def poisson_2_array(mu, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(poisson_2(mu))
    return rand_number_arr


def geometric_1_array(p, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(geometric_1(p))
    return rand_number_arr


def geometric_2_array(p, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(geometric_2(p))
    return rand_number_arr


def geometric_3_array(p, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(geometric_3(p))
    return rand_number_arr


def evaluate_expectation(rand_number_array):
    sum1 = 0.0
    for i in range(len(rand_number_array)):
        sum1 += rand_number_array[i]
    return sum1 / len(rand_number_array)


def evaluate_dispersion(rand_number_array, expect=-1234.0):
    sum1 = 0.0
    if expect == -1234.0:
        expect = evaluate_expectation(rand_number_array)
    for number in rand_number_array:
        sum1 += (number - expect) ** 2
    return sum1 / len(rand_number_array)


def distribution_function(rand_number_array, bins=10):
    counts, bin_edges = np.histogram(rand_number_array, bins=bins, density=False)
    cdf = np.cumsum(counts)
    cdf = cdf / cdf[-1]
    plt.bar(bin_edges[:-1], cdf, width=np.diff(bin_edges), align="edge", edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function (CDF)')
    plt.grid(True)
    plt.show()


def density_function(rand_number_array, bins = 10):
    plt.hist(rand_number_array, bins=bins, density=True, alpha=0.6, color='g', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Probability Density Function (PDF)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    input_value = -1
    rand_num_arr = list()
    table = PrettyTable()
    while input_value < 0 or input_value > 7:
        input_value = int(input("Enter an algorithm number:\n1 - Uniform\n2 - Binomial\n"
                                "3 - Geometric\n4 - Poisson\n"))
        if input_value == 1:
            rand_num_arr = uniform_distribution_array(1, 100, 10000)
            rand_num_arr.sort()
            expectation = evaluate_expectation(rand_num_arr)
            dispersion = evaluate_dispersion(rand_num_arr)

            table.field_names = ["Metric", "Value", "Deviation", "Theoretic"]
            table.add_row(["@M@", expectation, 50.5 - expectation, 50.5])
            table.add_row(["@D@", dispersion, 833.25 - dispersion, 833.25])

            distribution_function(rand_num_arr, np.arange(min(rand_num_arr), max(rand_num_arr) + 1) - 0.5)
            density_function(rand_num_arr, 10)
        elif input_value == 2:
            rand_num_arr = binomial_distribution_array(10, 0.5, 10000)
            rand_num_arr.sort()
            expectation = evaluate_expectation(rand_num_arr)
            dispersion = evaluate_dispersion(rand_num_arr)

            table.field_names = ["Metric", "Value", "Deviation", "Theoretic"]
            table.add_row(["@M@", expectation, 5.0 - expectation, 5.0])
            table.add_row(["@D@", dispersion, 2.5 - dispersion, 2.5])

            distribution_function(rand_num_arr)
            density_function(rand_num_arr)

        elif input_value == 3:
            rand_num_arr1 = geometric_1_array(0.5, 100000)
            rand_num_arr2 = geometric_2_array(0.5, 100000)
            rand_num_arr3 = geometric_3_array(0.5, 100000)

            rand_num_arr1.sort()
            rand_num_arr2.sort()
            rand_num_arr3.sort()

            expectation1 = evaluate_expectation(rand_num_arr1)
            dispersion1 = evaluate_dispersion(rand_num_arr1)

            expectation2 = evaluate_expectation(rand_num_arr2)
            dispersion2 = evaluate_dispersion(rand_num_arr2)

            expectation3 = evaluate_expectation(rand_num_arr3)
            dispersion3 = evaluate_dispersion(rand_num_arr3)

            table.field_names = ["Metric", "Value_1", "Value_2", "Value_3", "Theoretic"]
            table.add_row(["@M@", expectation1, expectation2, expectation3, 2.0])
            table.add_row(["@D@", dispersion1, dispersion2, dispersion3, 2.0])

            distribution_function(rand_num_arr1, np.arange(min(rand_num_arr1), max(rand_num_arr1) + 1) - 0.5)
            density_function(rand_num_arr1, np.arange(min(rand_num_arr1), max(rand_num_arr1) + 1) - 0.5)
            distribution_function(rand_num_arr2, np.arange(min(rand_num_arr2), max(rand_num_arr2) + 1) - 0.5)
            density_function(rand_num_arr2, np.arange(min(rand_num_arr1), max(rand_num_arr1) + 1) - 0.5)
            distribution_function(rand_num_arr3, np.arange(min(rand_num_arr3), max(rand_num_arr3) + 1) - 0.5)
            density_function(rand_num_arr3, np.arange(min(rand_num_arr1), max(rand_num_arr1) + 1) - 0.5)

        elif input_value == 4:
            rand_num_arr1 = poisson_1_array(10, 500000)
            rand_num_arr2 = poisson_2_array(10, 500000)

            rand_num_arr1.sort()
            rand_num_arr2.sort()

            expectation1 = evaluate_expectation(rand_num_arr1)
            dispersion1 = evaluate_dispersion(rand_num_arr1)

            expectation2 = evaluate_expectation(rand_num_arr2)
            dispersion2 = evaluate_dispersion(rand_num_arr2)

            table.field_names = ["Metric", "Value_1", "Value_2", "Theoretic"]
            table.add_row(["@M@", expectation1, expectation2, 10.0])
            table.add_row(["@D@", dispersion1, dispersion2, 10.0])

            distribution_function(rand_num_arr1, np.arange(min(rand_num_arr1), max(rand_num_arr1) + 1) - 0.5)
            density_function(rand_num_arr1, np.arange(min(rand_num_arr1), max(rand_num_arr1) + 1) - 0.5)
            distribution_function(rand_num_arr2, np.arange(min(rand_num_arr2), max(rand_num_arr2) + 1) - 0.5)
            density_function(rand_num_arr2, np.arange(min(rand_num_arr2), max(rand_num_arr2) + 1) - 0.5)

        else:
            print("Incorrect input value!")
    print(table)

