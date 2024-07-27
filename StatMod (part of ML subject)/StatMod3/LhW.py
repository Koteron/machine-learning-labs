import random
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import math


def uniform_distribution(min_val, max_val):
    return (max_val - min_val) * random.uniform(0, 1) + min_val


def uniform_distribution_array(min_val, max_val, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(uniform_distribution(min_val, max_val))
    return rand_number_arr


def normal_distribution_central():
    r = 0.0
    for _ in range(12):
        r += random.uniform(0, 1)
    return r - 6.0


def normal_central_array(amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(normal_distribution_central())
    return rand_number_arr


def normal_distribution_BoxMiller():
    return math.sqrt(-2 * math.log(random.uniform(0, 1))) *\
        math.cos(2 * math.pi * random.uniform(0, 1))


def normal_BoxMiller_array(amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(normal_distribution_BoxMiller())
    return rand_number_arr


def exponential_distribution(beta):
    return -beta*math.log(random.uniform(0, 1))


def exponential_distribution_array(beta, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(exponential_distribution(beta))
    return rand_number_arr


def chi_squared_distribution(N):
    YN = 0.0
    for _ in range(N):
        YN += normal_distribution_central() ** 2
    return YN


def chi_squared_distribution_array(N, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(chi_squared_distribution(N))
    return rand_number_arr


def student_distribution(N):
    return normal_distribution_central()/math.sqrt(chi_squared_distribution(N)/N)


def student_distribution_array(N, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(student_distribution(N))
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


def density_function(rand_number_array, bins=10):
    plt.hist(rand_number_array, bins=bins, density=True, alpha=0.6, color='g')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Probability Density Function (PDF)')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":

    TEST_AMOUNT = 10000

    input_value = -1
    rand_num_arr = list()
    table = PrettyTable()
    while input_value < 0 or input_value > 7:
        input_value = int(input("Enter an algorithm number:\n1 - Uniform\n2 - Normal\n"
                                "3 - Exponential\n4 - Chi-squared\n5 - Student\n"))
        if input_value == 1:
            rand_num_arr = uniform_distribution_array(1, 11, TEST_AMOUNT)
            rand_num_arr.sort()
            expectation = evaluate_expectation(rand_num_arr)
            dispersion = evaluate_dispersion(rand_num_arr)

            table.field_names = ["Metric", "Value", "Deviation", "Theoretic"]
            table.add_row(["@M@", expectation, 6.0 - expectation, 6.0])
            table.add_row(["@D@", dispersion, 8.3333 - dispersion, 8.3333])

            distribution_function(rand_num_arr, 50)
            density_function(rand_num_arr, 10)

        elif input_value == 2:
            rand_num_arr1 = normal_BoxMiller_array(TEST_AMOUNT)
            rand_num_arr2 = normal_central_array(TEST_AMOUNT)

            rand_num_arr1.sort()
            rand_num_arr2.sort()

            expectation1 = evaluate_expectation(rand_num_arr1)
            dispersion1 = evaluate_dispersion(rand_num_arr1)

            expectation2 = evaluate_expectation(rand_num_arr2)
            dispersion2 = evaluate_dispersion(rand_num_arr2)

            table.field_names = ["Metric", "Central", "Box-Miller", "Theoretic"]
            table.add_row(["@M@", expectation1, expectation2, 0.0])
            table.add_row(["@D@", dispersion1, dispersion2, 1.0])

            distribution_function(rand_num_arr1, 100)
            density_function(rand_num_arr1, 30)

            distribution_function(rand_num_arr2, 100)
            density_function(rand_num_arr2, 30)

        elif input_value == 3:
            rand_num_arr = exponential_distribution_array(1, TEST_AMOUNT)
            rand_num_arr.sort()
            expectation = evaluate_expectation(rand_num_arr)
            dispersion = evaluate_dispersion(rand_num_arr)

            table.field_names = ["Metric", "Value", "Theoretic"]
            table.add_row(["@M@", expectation, 1.0])
            table.add_row(["@D@", dispersion, 1.0])

            distribution_function(rand_num_arr, 100)
            density_function(rand_num_arr, 50)

        elif input_value == 4:
            rand_num_arr = chi_squared_distribution_array(10, TEST_AMOUNT)
            rand_num_arr.sort()
            expectation = evaluate_expectation(rand_num_arr)
            dispersion = evaluate_dispersion(rand_num_arr)

            table.field_names = ["Metric", "Value", "Theoretic"]
            table.add_row(["@M@", expectation, 10.0])
            table.add_row(["@D@", dispersion, 20.0])

            distribution_function(rand_num_arr, 100)
            density_function(rand_num_arr, 50)

        elif input_value == 5:
            rand_num_arr = student_distribution_array(10, TEST_AMOUNT)
            rand_num_arr.sort()
            expectation = evaluate_expectation(rand_num_arr)
            dispersion = evaluate_dispersion(rand_num_arr)

            table.field_names = ["Metric", "Value", "Deviation", "Theoretic"]
            table.add_row(["@M@", expectation, 0.0 - expectation, 0.0])
            table.add_row(["@D@", dispersion, 1.25 - dispersion, 1.25])

            distribution_function(rand_num_arr, 100)
            density_function(rand_num_arr, 50)
        else:
            print("Incorrect input value!")
    print(table)

