from collections import defaultdict


def main():
    k1, k2 = list(map(int, input().split()))

    sumRows = defaultdict(float)
    sumColumns = defaultdict(float)

    table = defaultdict(int)

    n = int(input())

    for i in range(n):
        x, y = list(map(int, input().split()))
        x -= 1
        y -= 1
        sumRows[x] += 1.0
        sumColumns[y] += 1.0
        table[(x, y)] += 1

    ans = 0.

    for k, v in table.items():
        expected = (sumRows[k[0]] * sumColumns[k[1]]) / n
        if expected == 0:
            continue
        ans += ((v - expected) ** 2) / expected - expected
    print(ans + n)


if __name__ == '__main__':
    main()
