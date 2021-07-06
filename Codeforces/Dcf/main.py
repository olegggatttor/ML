import random
import numpy


class Point:
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
    df = lambda w_j, point: 2 * (ys_predicted - point.y) * point.xs[j] + 2*tau*ws[w_j]
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
    ws = [random.uniform(-1 / (2 * n), 1 / (2 * n)) for _ in range(m)]
    step = 1e-3
    for i in range(3, 50000):
        print(ws)
        new_ws = SMAPE_diff(step, ws, points)
        ws = [w - n_w for w, n_w in zip(ws, new_ws)]
        # step = 1 / i
    # ys_predicted = [sum([w * x for w, x in zip(ws, pt.xs)]) for pt in points]
    for w in ws:
        print(w)


if __name__ == '__main__':
    main()
