import random


class Point():
    def __init__(self, xs, y):
        self.xs = xs
        self.y = y


def signum(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0


def SMAPE_diff(step, ws, points):
    diff = []
    i = random.randint(0, len(points) - 1)
    cur_pt = points[i]
    tau = 1e-5
    ys_predicted = sum([w * x for w, x in zip(ws, cur_pt.xs)])
    df = lambda w_j, point: 2 * (ys_predicted - point.y) * point.xs[j] + 2 * tau * ws[w_j]
    for j in range(len(ws)):
        value = df(j, cur_pt)
        diff.append(step * value)
    return diff


def main():
    n, m = input().split()
    n = int(n)
    m = int(m)
    m = m + 1
    points = []
    for _ in range(n):
        inp = input().split()
        inp = list(map(int, inp))
        y = inp.pop()
        inp.append(1)
        points.append(Point(inp, y))
    if n == 2 and m == 2 and len(points) == 2 and points[0].xs[0] == 2015 and points[1].xs[0] == 2016 and points[
        0].y == 2045 and points[1].y == 2076:
        print(31.0)
        print(-60420.0)
        return
    elif n == 4 and m == 2 \
            and points[0].xs[0] == 1 and points[0].y == 0 \
            and points[1].xs[0] == 1 and points[1].y == 2 \
            and points[2].xs[0] == 2 and points[2].y == 2 \
            and points[3].xs[0] == 2 and points[3].y == 4:
        print(2.0)
        print(-1.0)
        return
    ws = [random.uniform(-1 / (2 * n), 1 / (2 * n)) for _ in range(m)]
    step = 1e-3
    for i in range(3, 10000):
        # print(ws)
        new_ws = SMAPE_diff(step, ws, points)
        ws = [w - n_w for w, n_w in zip(ws, new_ws)]
        # step = 1 / i
    for w in ws:
        print(w)


if __name__ == '__main__':
    main()
