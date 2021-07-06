import operator
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from math import log, sqrt, exp


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
        for _ in range(int(sqrt(len(points[0].xs)))):
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
        data = list(map(float, row))
        y = int(data.pop())
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


def count_weighted_loss(points, points_weights, tree):
    loss = 0
    for point, w in zip(points, points_weights):
        y_predicted = predict(point, tree)
        if point.y != y_predicted:
            loss += w
    return loss


def classify(point, ws, classifiers):
    res = sum([w * predict(point, cl) for w, cl in zip(ws, classifiers)])
    if res > 0:
        return 1
    elif res < 0:
        return -1
    else:
        return 0


def count_accuracy(points, coefficients, classifiers):
    accuracy = 0
    for point in points:
        ys_predicted = classify(point, coefficients, classifiers)
        if ys_predicted == point.y:
            accuracy += 1
    return accuracy / len(points)


def boost(points, forest):
    MAX_ITERS = 56
    points_weights = [1 / len(points)] * len(points)
    classifiers_coefficients = []
    classifiers = []

    min_x = min(points, key=lambda pt: pt.xs[0]).xs[0]
    max_x = max(points, key=lambda pt: pt.xs[0]).xs[0]
    min_y = min(points, key=lambda pt: pt.xs[1]).xs[1]
    max_y = max(points, key=lambda pt: pt.xs[1]).xs[1]

    xs_init_good = [pt.xs[0] for pt in points if pt.y == 1]
    ys_init_good = [pt.xs[1] for pt in points if pt.y == 1]

    xs_init_bad = [pt.xs[0] for pt in points if pt.y == -1]
    ys_init_bad = [pt.xs[1] for pt in points if pt.y == -1]

    xs_test = np.linspace(min_x, max_x, 50)
    ys_test = np.linspace(min_y, max_y, 50)

    xs_algorithm_accuracy = []
    ys_algorithm_accuracy = []

    for t in range(1, MAX_ITERS):
        # print(t)
        weighted_losses = [count_weighted_loss(points, points_weights, tree) for tree in forest]
        # print(weighted_losses)
        # print(weighted_losses)
        min_weighted_loss = min(weighted_losses)
        weak_classifier = forest[weighted_losses.index(min_weighted_loss)]
        classifier_k = log((1 - min_weighted_loss) / min_weighted_loss)

        classifiers.append(weak_classifier)
        classifiers_coefficients.append(classifier_k)

        points_weights = [old_w * exp(-classifier_k * pt.y * predict(pt, weak_classifier)) for pt, old_w in
                          zip(points, points_weights)]
        normalizer = sum(points_weights)
        points_weights = [w / normalizer for w in points_weights]

        xs_algorithm_accuracy.append(t)
        ys_algorithm_accuracy.append(count_accuracy(points, classifiers_coefficients, classifiers))

        if t in [1, 2, 3, 5, 8, 13, 34, 55]:

            for i in range(len(xs_test)):
                for j in range(len(ys_test)):
                    test_pt = Point([xs_test[i], ys_test[j]], -1)
                    predicted_y = classify(test_pt, classifiers_coefficients, classifiers)
                    color = 'w'
                    if predicted_y < 0:
                        color = 'r'
                    elif predicted_y > 0:
                        color = 'b'
                    plt.scatter(xs_test[i], ys_test[j], c=color, alpha=0.2)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.scatter(xs_init_good, ys_init_good, c='b')
            plt.scatter(xs_init_bad, ys_init_bad, c='r')
            plt.title('Classification on ' + str(t) + ' boost iteration.')
            plt.show()

    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.title('Accuracy on every iteration of boosting.')
    plt.plot(xs_algorithm_accuracy, ys_algorithm_accuracy)
    plt.show()


def init_trees_and_boost(points, max_tree_height):
    forest = []
    for h in [max_tree_height] * 100:
        global cur_pos
        cur_pos = 0
        attributes = list(range(len(points[0].xs)))
        random_chips = random.sample(points, random.randint(int(len(points) / 3),
                                                            2 * int(len(points) / 3)))
        while len(random_chips) < len(points):
            random_chips.append(random_chips[random.randint(0, len(random_chips) - 1)])

        ys = count_marks(random_chips)

        size, tree = build_tree(random_chips, h, attributes,
                                get_entropy(ys, len(random_chips)), len(random_chips), False)

        forest.append(tree)
    boost(points, forest)


def main():
    chips_dataframe = pd.read_csv('chips.csv')
    chips_dataframe['class'] = chips_dataframe['class'].apply(lambda x: 1 if x == 'P' else -1)
    chips_dataframe = chips_dataframe.sample(frac=1)
    chips_points = get_points(chips_dataframe)
    init_trees_and_boost(chips_points, 4)

    geyser_dataframe = pd.read_csv('geyser.csv')
    geyser_dataframe['class'] = geyser_dataframe['class'].apply(lambda x: 1 if x == 'P' else -1)
    geyser_dataframe = geyser_dataframe.sample(frac=1)
    geyser_points = get_points(geyser_dataframe)
    init_trees_and_boost(geyser_points, 7)


if __name__ == '__main__':
    main()
