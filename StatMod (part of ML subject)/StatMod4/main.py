import math
import random
from threading import Thread


def tester(L, N):
    n = [2, 3, 3]
    m = 3
    lamb = [40, 10, 80]

    d = 0
    for _ in range(0, N):
        x = []
        for i in range(0, m):
            t = []
            for j in range(0, n[i]):
                t.append(-math.log(random.uniform(0, 1)) / (lamb[i] * pow(10, -6)))

            for j in range(0, L[i]):
                ind = min(range(len(t)), key=t.__getitem__)
                t[ind] -= math.log(random.uniform(0, 1)) / (lamb[i] * pow(10, -6))

            for j in range(0, n[i]):
                x.append(t[j])

        if not system_working(x, 8760):
            d = d + 1

    return 1 - d / N


def run(i):
    P0 = 0.99
    L = [0, 0, 0]
    L[0] = i
    for j in range(1, 4):
        L[1] = j
        for k in range(1, 4):
            L[2] = k
            P = tester(L, round(2.5758 ** 2 * 0.99 * (1-0.99) / 0.0003 / 0.0003))
            print("p=",P)
            if P > P0:
                print("P={0}, L={1}, L_sum={2}".format(P, L, sum(L)))


def system_working(x, T):
    return (x[0] > T or x[1] > 0) and \
            x[2] > T and x[3] > T and x[4] > T and \
            (x[5] > T or x[6] > T or x[7] > T)


if __name__ == "__main__":
    threads = []
    for i in range(1, 4):
        x = Thread(target=run, args=(i,))
        x.start()
        threads.append(x)

    for th in threads:
        th.join()

