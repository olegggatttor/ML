import operator
import pandas as pd
import random
import matplotlib.pyplot as plt
from math import log, sqrt


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


def build_tree(points, h, avaliable_xs, entropy, amount_of_elements, is_forest):
    global cur_pos
    ys = count_marks(points)
    if (h == 0 and not is_forest) or len(ys) == 1:
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
    copy_attrs = avaliable_xs.copy()
    if is_forest:
        attrs_for_random_tree = []
        for _ in range(int(sqrt(len(points[0].xs)) + 1)):
            if len(copy_attrs) != 0:
                pos = random.randint(0, len(copy_attrs) - 1)
                attrs_for_random_tree.append(copy_attrs.pop(pos))
        copy_attrs = attrs_for_random_tree
    for x in copy_attrs:
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
        cur_IG = entropy - (len(left) / amount_of_elements) * l_entropy - (len(right) / amount_of_elements) * r_entropy
        if cur_IG > IG:
            IG = cur_IG
            best_l = left
            best_r = right
            best_x = x
            best_avg = avg
            best_l_entropy = l_entropy
            best_r_entropy = r_entropy
    if len(best_l) == 0 or len(best_r) == 0:
        leaf_class = max(ys.items(), key=operator.itemgetter(1))[0]
        my_pos = cur_pos
        cur_pos += 1
        return 1, Node(None, None, leaf_class, True, my_pos)
    my_pos = cur_pos
    cur_pos += 1
    l_size, l = build_tree(best_l, h - 1, avaliable_xs, best_l_entropy, len(best_l), is_forest)
    r_size, r = build_tree(best_r, h - 1, avaliable_xs, best_r_entropy, len(best_r), is_forest)
    return (l_size + r_size + 1), Node(l, r, best_avg, False, my_pos, best_x)


def traverse_print(tree):
    if tree.is_leaf:
        print('C ' + str(tree.param))
    else:
        print('Q ' + str(tree.best_x + 1) + ' ' + str(tree.param) + ' ' + str(tree.l.pos) + ' ' + str(tree.r.pos))
        traverse_print(tree.l)
        traverse_print(tree.r)


def get_points(dataframe):
    points = []
    for index, row in dataframe.iterrows():
        data = list(map(int, row))
        y = data.pop()
        points.append(Point(data, y))
    return points


class Result:
    def __init__(self, loss, tree, data_pos, height):
        self.loss = loss
        self.tree = tree
        self.data_pos = data_pos
        self.height = height


def predict(point, tree):
    if tree.is_leaf:
        return tree.param
    else:
        decision_attr = tree.best_x
        if point.xs[decision_attr] < tree.param:
            return predict(point, tree.l)
        else:
            return predict(point, tree.r)


def test_tree(points, tree):
    loss = 0
    for point in points:
        y_predicted = predict(point, tree)
        if point.y != y_predicted:
            loss += 1
    return loss


def forest_predict(points, forest):
    forest_loss = 0
    for pt in points:
        predictions = [predict(pt, tree) for tree in forest]
        y_predicted = predictions[0]
        times_predicted = predictions.count(predictions[0])
        for mark in set(predictions):
            cur_pred_count = predictions.count(mark)
            if cur_pred_count > times_predicted:
                times_predicted = cur_pred_count
                y_predicted = mark
        if y_predicted != pt.y:
            forest_loss += 1
    forest_loss /= len(points)
    return forest_loss


def build_plot(result, type):
    file_pos = result.data_pos
    best_h = result.height
    best_loss = result.loss

    print(type)
    print("Best height: " + str(best_h))
    print("Lowest loss: " + str(best_loss))

    test_dataframe = pd.read_csv("data/" + file_pos + "_test.csv")
    train_dataframe = pd.read_csv("data/" + file_pos + "_train.csv")

    points_test = get_points(test_dataframe)
    points_train = get_points(train_dataframe)

    ys_train_amounts = count_marks(points_train)
    attributes = list(range(len(points_train[0].xs)))
    xs = []
    ys_train = []
    ys_test = []
    for h in range(1, 21):
        size, tree = build_tree(points_train, h, attributes,
                                get_entropy(ys_train_amounts, len(points_train)), len(points_train), False)
        loss_test = 1 - test_tree(points_test, tree) / len(points_test)
        loss_train = 1 - test_tree(points_train, tree) / len(points_train)

        xs.append(h)
        ys_train.append(loss_train)
        ys_test.append(loss_test)

    plt.plot(xs, ys_train, label="train")
    plt.plot(xs, ys_test, label="test")
    plt.title(type)
    plt.xlabel("height")
    plt.ylabel("percentage of right predictions")
    plt.legend()
    plt.show()


def main():
    heights_to_res = {}
    for current_file in range(1, 22):
        print("Current file:" + str(current_file))
        file_pos = "0" if int(current_file / 10) == 0 else ""
        file_pos += str(current_file)
        train_dataframe = pd.read_csv("data/" + file_pos + "_train.csv")
        test_dataframe = pd.read_csv("data/" + file_pos + "_test.csv")

        points_train = get_points(train_dataframe)
        points_test = get_points(test_dataframe)

        ys_train_amounts = count_marks(points_train)

        forest = []
        best_h = -1
        best_loss = 100
        for h in range(1, 21):
            attributes = list(range(len(points_train[0].xs)))
            heights_to_res.setdefault(h, [])
            size, tree = build_tree(points_train, h, attributes,
                                    get_entropy(ys_train_amounts, len(points_train)), len(points_train), False)
            loss = test_tree(points_test, tree) / len(points_test)
            if loss < best_loss:
                best_h = h
                best_loss = loss
            res = Result(loss, tree, file_pos, h)
            heights_to_res[h].append(res)

        trees_in_forest = 100
        for _ in range(trees_in_forest):
            # forest build
            attributes = list(range(len(points_train[0].xs)))
            train_sample = random.sample(points_train, int(len(points_train) / 4))
            while len(train_sample) < len(points_train):
                train_sample.append(train_sample[random.randint(0, len(train_sample) - 1)])

            ys_train_sample_amounts = count_marks(train_sample)

            _, tree_for_forest = build_tree(train_sample, best_h, attributes,
                                            get_entropy(ys_train_sample_amounts, len(train_sample)), len(train_sample),
                                            True)
            forest.append(tree_for_forest)

        forest_loss_train = forest_predict(points_train, forest)
        forest_loss_test = forest_predict(points_test, forest)
        print(">Best accuracy on single tree with height " + str(best_h) + ": " + str(1 - best_loss))
        print(">Forest accuracy on train: " + str(1 - forest_loss_train))
        print(">Forest accuracy on test: " + str(1 - forest_loss_test))
        print()

    optimals = []
    for i in range(0, 21):
        cur = []
        for h in range(1, 21):
            cur.append(heights_to_res[h][i])
        opt_for_cur = min(cur, key=lambda x: x.loss)
        optimals.append(opt_for_cur)
    optimals = sorted(optimals, key=lambda x: x.height)
    best_small = optimals[0]
    best_high = optimals[len(optimals) - 1]

    build_plot(best_small, "small")
    build_plot(best_high, "high")


if __name__ == '__main__':
    main()
