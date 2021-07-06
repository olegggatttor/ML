import operator
from math import log


class Point:
    def __init__(self, xs, y):
        self.xs = xs
        self.y = y


class Node:
    def __init__(self, l, r, param, is_leaf, pos, best_x=-1):
        self.l = l
        self.r = r
        self.param = param
        self.is_leaf = is_leaf
        self.best_x = best_x
        self.pos = pos


def count_marks(points):
    ys = dict()
    for pt in points:
        ys.setdefault(pt.y, 0)
        ys[pt.y] += 1
    return ys


def get_entropy(ys_map, amount_of_classes):
    return -sum([(ys_map[key] / amount_of_classes) * log((ys_map[key] / amount_of_classes), 2) for key in ys_map])


cur_pos = 1


def build_tree(points, h, avaliable_xs, entropy, amount_of_elements):
    global cur_pos
    ys = count_marks(points)
    if h == 0 or len(ys) == 1:
        leaf_class = max(ys.items(), key=operator.itemgetter(1))[0]
        my_pos = cur_pos
        cur_pos += 1
        return 1, Node(None, None, leaf_class, True, my_pos)
    best_l = []
    best_r = []
    best_l_entropy = 0
    best_r_entropy = 0
    best_avg = 0
    best_x = -1
    IG = 0
    for x in avaliable_xs:
        avg = sum(list(map(lambda pt: pt.xs[x], points))) / len(points)
        left = []
        right = []
        for pt in points:
            if pt.xs[x] < avg:
                left.append(pt)
            else:
                right.append(pt)
        l_map = count_marks(left)
        r_map = count_marks(right)
        l_entropy = get_entropy(l_map, len(left))
        r_entropy = get_entropy(r_map, len(right))
        cur_IG = entropy - (len(left) / amount_of_elements) * l_entropy - (len(right) /amount_of_elements) * r_entropy
        if cur_IG > IG:
            IG = cur_IG
            best_l = left
            best_r = right
            best_x = x
            best_avg = avg
            best_l_entropy = l_entropy
            best_r_entropy = r_entropy
    if len(best_l) == 0 or len(best_r) == 0 or IG == 0:
        leaf_class = max(ys.items(), key=operator.itemgetter(1))[0]
        my_pos = cur_pos
        cur_pos += 1
        return 1, Node(None, None, leaf_class, True, my_pos)
    my_pos = cur_pos
    cur_pos += 1
    l_size, l = build_tree(best_l, h - 1, avaliable_xs, best_l_entropy, amount_of_elements)
    r_size, r = build_tree(best_r, h - 1, avaliable_xs, best_r_entropy, amount_of_elements)
    return (l_size + r_size + 1), Node(l, r, best_avg, False, my_pos, best_x)


def traverse_print(tree):
    if tree.is_leaf:
        print('C ' + str(tree.param))
    else:
        print('Q ' + str(tree.best_x + 1) + ' ' + str(tree.param) + ' ' + str(tree.l.pos) + ' ' + str(tree.r.pos))
        traverse_print(tree.l)
        traverse_print(tree.r)


def main():
    m, k, h = list(map(int, input().split()))
    n = int(input())
    points = []
    ys = set()
    for i in range(n):
        data = list(map(int, input().split()))
        y = data.pop()
        ys.add(y)
        points.append(Point(data, y))
    ys_map = count_marks(points)
    size, tree = build_tree(points, h, list(range(m)), get_entropy(ys_map, n), n)
    print(size)
    traverse_print(tree)


if __name__ == '__main__':
    main()
