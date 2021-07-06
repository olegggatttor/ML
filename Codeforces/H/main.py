def activate(value):
    return 1 if value > 0 else 0


def gen_table(m):
    table = []
    for i in range(2 ** m):
        table.append("{0:b}".format(i))
        while len(table[i]) < m:
            table[i] = '0' + table[i]
        table[i] = table[i][::-1]
    sorted(table)
    return table


def main():
    m = int(input())
    results = [float(input()) for _ in range(2 ** m)]
    ones = int(sum(results))
    if ones == 0:
        print(1)
        print(1)
        print(' '.join([str(0.0)] * m), end=' ')
        print(-1.0)
        return

    print(2)
    print(str(ones) + ' 1')
    table = gen_table(m)
    for i, row in zip(range(2 ** m), table):
        if results[i] == 1.0:
            for x_i in row:
                if x_i == '1':
                    print(1.0, end=' ')
                else:
                    print(-1.0, end=' ')
            print(0.5 - row.count('1'))
    print(' '.join([str(1.0)] * ones), end=' ')
    print(-0.5)


if __name__ == '__main__':
    main()
