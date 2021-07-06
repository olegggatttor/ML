from math import log, exp


class Message:
    def __init__(self, c, words):
        self.c = c
        self.words = words


class NaiveBayesClassifier:

    def __init__(self, k, a, lambdas, train_msgs):
        self.amount_of_classes = k
        self.error_lambdas = lambdas
        self.alpha = a
        self.train_msgs = train_msgs
        self.amount_of_msgs = len(train_msgs)
        self.p_of_class = []
        self.all_words = set()
        self.word_to_docs = dict()
        probs_of_classes_temp = [0] * k
        for msg in train_msgs:
            probs_of_classes_temp[msg.c] += 1
            for word in msg.words:
                self.word_to_docs.setdefault((msg.c, word), 0)
                self.word_to_docs[(msg.c, word)] += 1
                self.all_words.add(word)
        self.p_word_class = []
        for cl in range(k):
            self.p_of_class.append(probs_of_classes_temp[cl] / self.amount_of_msgs)
            for word in self.all_words:
                self.p_word_class.append({})
                self.p_word_class[cl][word] = self.p(cl, word)

    def p(self, cl, word):
        amount = self.word_to_docs.get((cl, word), 0)
        return (amount + self.alpha) / ((self.p_of_class[cl] * self.amount_of_msgs) + 2 * self.alpha)

    def count_probs(self, test):
        results1 = [0] * self.amount_of_classes
        results2 = [0] * self.amount_of_classes
        ya_russkiy = [0] * self.amount_of_classes

        for cl in range(self.amount_of_classes):
            results1[cl] = self.p_of_class[cl] * self.error_lambdas[cl]
            results2[cl] = self.p_of_class[cl] * self.error_lambdas[cl]
            for word in self.all_words:
                if results1[cl] < 1e-200:
                    results1[cl] *= 1e+200
                    ya_russkiy[cl] += 1
                if word in test.words:
                    results1[cl] *= self.p_word_class[cl][word]
                    results2[cl] *= self.p_word_class[cl][word]
                else:
                    results1[cl] *= (1 - self.p_word_class[cl][word])
                    results2[cl] *= (1 - self.p_word_class[cl][word])
        bot = sum(results2)
        if bot != 0:
            results = [el / bot for el in results2]
            return results
        max_russkiy = max(ya_russkiy)
        for i in range(len(ya_russkiy)):
            while ya_russkiy[i] < max_russkiy:
                results1[i] *= 1e+200
                ya_russkiy[i] += 1
        bot = sum(results1)
        results = [el / bot for el in results1]
        return results


def main():
    k = int(input())
    lambdas = list(map(int, input().split()))
    alpha = int(input())
    N = int(input())
    train = []
    for i in range(N):
        msg = input().split()
        train.append(Message(int(msg[0]) - 1, list(set(msg[2:]))))
    tester = NaiveBayesClassifier(k, alpha, lambdas, train)
    M = int(input())
    for i in range(M):
        h, *t = input().split()
        ans = tester.count_probs(Message(-1, list(set(t))))
        print(" ".join(list(map(str, ans))))


if __name__ == '__main__':
    main()
