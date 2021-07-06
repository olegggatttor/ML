def main():
    from collections import defaultdict
    from math import log

    kx, ky = list(map(int, input().split()))

    n = int(input())

    xToYs = defaultdict(lambda: defaultdict(float))
    xGivenY = defaultdict(float)
    amountOfXs = defaultdict(float)

    for i in range(n):
        x, y = list(map(int, input().split()))
        x -= 1
        xToYs[x][y] += 1.
        amountOfXs[x] += 1. / n

    for (x, amountsOfYs) in xToYs.items():
        amnt = float(sum(amountsOfYs.values()))
        p = list(map(lambda y: -(y / amnt) * log(y / amnt), amountsOfYs.values()))
        xGivenY[x] = sum(p)

    entropy = 0.

    for x in range(kx):
        entropy += amountOfXs[x] * xGivenY[x]

    print(entropy) # 0.4852030263919617

if __name__ == '__main__':
    main()

