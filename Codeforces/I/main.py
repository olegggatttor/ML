from math import tanh, cosh


class Action:
    def __init__(self, action, params):
        self.action = action
        self.params = params


def apply_to_matrix(func, matrix):
    new_matrix = []
    for row in matrix:
        new_matrix.append(list(map(func, row)))
    return new_matrix


def mul(matrix_1, matrix_2):
    n = len(matrix_1)
    k = len(matrix_1[0])
    m = len(matrix_2[0])
    new_matrix = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for t in range(k):
                new_matrix[i][j] += (matrix_1[i][t] * matrix_2[t][j])
    return new_matrix


def eval_matrix_op(op, matrices):
    rows = len(matrices[0])
    columns = len(matrices[0][0])
    new_matrix = []
    for i in range(rows):
        row = matrices[0][i].copy()
        for cur in range(1, len(matrices)):
            for j in range(columns):
                row[j] = op(row[j], matrices[cur][i][j])
        new_matrix.append(row)
    return new_matrix


def main():
    n, m, k = list(map(int, input().split()))
    actions = []
    for i in range(n):
        inp = input().split()
        action = inp.pop(0)
        actions.append(Action(action, list(map(int, inp))))
    vertices = []
    for m_pos in range(m):
        n_cur, m_cur = actions[m_pos].params
        matrix = []
        for i in range(n_cur):
            matrix.append(list(map(float, input().split())))
        vertices.append(matrix)

    diffs = []
    for vert in vertices:
        diffs.append([
            [0.0] * len(vert[0]) for _ in range(len(vert))
        ])
    for i, action in zip(range(m, n), actions[m:]):
        if action.action == 'tnh':
            arg = vertices[action.params[0] - 1]
            vertices.append(apply_to_matrix(tanh, arg))
        elif action.action == 'rlu':
            alpha = action.params[0]
            arg = vertices[action.params[1] - 1]
            vertices.append(apply_to_matrix(lambda x: x / alpha if x < 0 else x, arg))
        elif action.action == 'mul':
            arg1 = vertices[action.params[0] - 1]
            arg2 = vertices[action.params[1] - 1]
            vertices.append(mul(arg1, arg2))
        elif action.action == 'sum':
            matrices = list(map(lambda x: vertices[x - 1], action.params[1:]))
            vertices.append(eval_matrix_op(lambda x, y: x + y, matrices))
        elif action.action == 'had':
            matrices = list(map(lambda x: vertices[x - 1], action.params[1:]))
            vertices.append(eval_matrix_op(lambda x, y: x * y, matrices))
        if i < n - k:
            diffs.append([
                [0.0] * len(vertices[-1][0]) for _ in range(len(vertices[-1]))
            ])
    answers = vertices[-k:]
    for ans in answers:
        diffs.append([list(map(float, input().split())) for _ in range(len(ans))])
        for row in ans:
            print(' '.join(list(map(str, row))))

    for diff_i, action in zip(range(n - 1, m - 1, -1), list(reversed(actions[m:]))):
        if action.action == 'tnh':
            pos = action.params[0]
            thn_diff = apply_to_matrix(lambda x: 1.0 / (cosh(x) ** 2), vertices[pos - 1])
            diffs[pos - 1] = eval_matrix_op(lambda x, y: x + y, [diffs[pos - 1],
                                                                 eval_matrix_op(lambda x, y: x * y,
                                                                                [diffs[diff_i], thn_diff])])
        elif action.action == 'rlu':
            alpha = action.params[0]
            pos = action.params[1]
            rlu_diff = apply_to_matrix(lambda x: 1.0 / alpha if x < 0 else 1.0, vertices[pos - 1])
            diffs[pos - 1] = eval_matrix_op(lambda x, y: x + y, [diffs[pos - 1],
                                                                 eval_matrix_op(lambda x, y: x * y,
                                                                                [diffs[diff_i], rlu_diff])])
        elif action.action == 'mul':
            pos1 = action.params[0]
            pos2 = action.params[1]
            transposed_vert1 = [[vertices[pos1 - 1][j][i] for j in range(len(vertices[pos1 - 1]))] for i in
                                range(len(vertices[pos1 - 1][0]))]
            transposed_vert2 = [[vertices[pos2 - 1][j][i] for j in range(len(vertices[pos2 - 1]))] for i in
                                range(len(vertices[pos2 - 1][0]))]
            diffs[pos1 - 1] = eval_matrix_op(lambda x, y: x + y,
                                             [diffs[pos1 - 1], mul(diffs[diff_i], transposed_vert2)])
            diffs[pos2 - 1] = eval_matrix_op(lambda x, y: x + y,
                                             [diffs[pos2 - 1], mul(transposed_vert1, diffs[diff_i])])
        elif action.action == 'sum':
            for mx_pos in action.params[1:]:
                diffs[mx_pos - 1] = eval_matrix_op(lambda x, y: x + y, [diffs[mx_pos - 1], diffs[diff_i]])
        elif action.action == 'had':
            matrices = list(map(lambda x: vertices[x - 1], action.params[1:]))
            for i, mx_pos in zip(range(len(matrices)), action.params[1:]):
                save = matrices.pop(i)
                matrices.append(diffs[diff_i])
                diffs[mx_pos - 1] = eval_matrix_op(lambda x, y: x + y,
                                                   [diffs[mx_pos - 1], eval_matrix_op(lambda x, y: x * y, matrices)])
                matrices = matrices[:-1]
                matrices.insert(i, save)
    for i in range(m):
        for row in diffs[i]:
            print(' '.join(list(map(str, row))))


if __name__ == '__main__':
    print(apply_to_matrix(lambda x: x / 8 if x < 0 else x, [[10, 0], [0, -2]]))
    # main()
