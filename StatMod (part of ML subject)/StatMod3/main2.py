import random
import LhW as lhw
from prettytable import PrettyTable
import math
import numpy as np
from matplotlib import pyplot as plt

def kolmogorov_critical(alpha):
    return math.sqrt(-0.5 * math.log((1-alpha)/2))


def rayleigh_distribution(sigma):
    # return sigma*math.sqrt(2*random.uniform(0, 1))
    return math.sqrt(-2.0 * sigma ** 2 * math.log(1 - random.uniform(0, 1)))


def rayleigh_distribution_cdf(x, sigma):
    # return sigma*math.sqrt(2*random.uniform(0, 1))
    return 1 - math.exp(-x ** 2 / (2 * sigma ** 2))


def rayleigh_distribution_array(sigma, amount):
    rand_number_arr = list()
    for i in range(amount):
        rand_number_arr.append(rayleigh_distribution(sigma))
    return rand_number_arr


def kolmogorov_criteria(rand_number_arr, sigma):
    ecdf = np.arange(1, len(rand_number_arr) + 1) / len(rand_number_arr)
    ks_statistic = -1
    for i in range(len(rand_number_arr)):
        ks_statistic = max(ks_statistic,
                           abs(ecdf[i] - rayleigh_distribution_cdf(rand_number_arr[i], sigma)))
    return ks_statistic


if __name__ == "__main__":
    TEST_AMOUNT = 100
    KOLM_TEST_AMOUNT = 50
    SIGMA = 1
    ALPHA = 0.05

    table = PrettyTable()
    rand_num_arr = rayleigh_distribution_array(SIGMA, TEST_AMOUNT)
    rand_num_arr.sort()
    expectation = lhw.evaluate_expectation(rand_num_arr)
    dispersion = lhw.evaluate_dispersion(rand_num_arr)

    table.field_names = ["Metric", "Value"]
    table.add_row(["@M@", expectation])
    table.add_row(["@D@", dispersion])
    print(table)

    lhw.distribution_function(rand_num_arr, 0.05)
    lhw.density_function(rand_num_arr, 0.1)

    calc = list()
    crit = list()
    calc.append(kolmogorov_criteria(rand_num_arr, SIGMA))
    crit.append(kolmogorov_critical(ALPHA))
    for _ in range(KOLM_TEST_AMOUNT-1):
        rand_num_arr = rayleigh_distribution_array(SIGMA, TEST_AMOUNT)
        rand_num_arr.sort()
        calc.append(kolmogorov_criteria(rand_num_arr, SIGMA))
        crit.append(kolmogorov_critical(ALPHA))

    plt.bar([i - 0.25 for i in range(KOLM_TEST_AMOUNT)], calc, width=0.5, align='center')
    plt.bar([i + 0.25 for i in range(KOLM_TEST_AMOUNT)], crit, width=0.5,
            align='center')
    plt.show()
