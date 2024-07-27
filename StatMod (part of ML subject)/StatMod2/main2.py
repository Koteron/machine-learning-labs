import math
import LhD as lhd
from scipy.stats import chi2
from matplotlib import pyplot as plt

if __name__ == "__main__":

    # Test Parameters
    N = 5
    M = 4
    SIGNIFICANCE_LEVEL = 0.05
    TEST_AMOUNT = 50

    chi_squared_list = list()
    chi_critical_list = list()

    for _ in range(TEST_AMOUNT):
        random_number_array = lhd.uniform_distribution_array(0, 2**M-1, N)
        random_number_array.sort()
        print("random number array: ", random_number_array)
        spacings = list()
        for i in range(1, len(random_number_array)):
            spacings.append(random_number_array[i] - random_number_array[i-1])
        spacings.append(random_number_array[0] + 2**M - random_number_array[-1])
        print("spacings: ", spacings)
        spacings_copy = spacings.copy()
        n_list = list()
        for i in range(len(spacings)):
            n_list.append(0)
        while len(spacings_copy) != 0:
            element = spacings_copy[0]
            n_list[spacings_copy.count(element)-1] += 1
            i = 0
            while i < len(spacings_copy):
                if spacings_copy[i] == element:
                    spacings_copy.remove(spacings_copy[i])
                else:
                    i += 1
        print("n_list: ", n_list)

        lambda_value = N**3/4/2**M

        p_list = list()
        for i in range(len(n_list)):
            if n_list[i] != 0:
                p_list.append(lambda_value**i*math.exp(-lambda_value)/math.factorial(i))

        print("p_list: ", p_list)

        j = 0
        chi_squared_value = 0.0
        for i in range(len(n_list)):
            if n_list[i] != 0:
                chi_squared_value += (n_list[i] - N * p_list[j]) ** 2 / (N * p_list[j])
                j += 1
        if p_list.count(0) > 0:
            chi_squared_value += (0 - N * (1 - sum(p_list))) ** 2 / (N * (1 - sum(p_list)))

        print("Chi-squared: ", chi_squared_value)
        critical_chi = chi2.ppf(1-SIGNIFICANCE_LEVEL, len(p_list)+1)
        print("Critical chi-squared: ", critical_chi)
        chi_squared_list.append(chi_squared_value)
        chi_critical_list.append(critical_chi)

    plt.bar([i-0.25 for i in range(TEST_AMOUNT)], chi_squared_list, width=0.5, align='center')
    plt.bar([i+0.25 for i in range(TEST_AMOUNT)], chi_critical_list, width=0.5, align='center')
    plt.show()



