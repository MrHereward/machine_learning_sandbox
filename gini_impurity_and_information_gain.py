import matplotlib.pyplot as plt
import numpy as np

def gini_impurity(labels):
    if not labels:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)

def entropy(labels):
    if not labels:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return -np.sum(fractions * np.log2(fractions))

criterion_function = {
    'gini': gini_impurity,
    'entropy': entropy
}

def weighted_impurity(groups, criterion='gini'):
    """
    :param groups: lista wezlow potomnych, z ktorych kazdy zawiera liste etykiet klas
    :param criterion: wskaznik jakosci podzialu
    :return: float, weighted impurity
    """
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * criterion_function[criterion](group)
    return weighted_sum

print(f'{gini_impurity([1, 1, 0, 1, 0]):.4f}')
print(f'{gini_impurity([1, 1, 0, 1, 0, 0]):.4f}')
print(f'{gini_impurity([1, 1, 1, 1]):.4f}')

print(f'{entropy([1, 1, 0, 1, 0]):.4f}')
print(f'{entropy([1, 1, 0, 1, 0, 0]):.4f}')
print(f'{entropy([1, 1, 1, 1]):.4f}')

children_1 = [[1, 0, 1], [0, 1]]
children_2 = [[1, 1], [0, 0, 1]]
print(f'Entropia 1: {weighted_impurity(children_1, 'entropy'):.4f}')
print(f'Entropia 2: {weighted_impurity(children_2, 'entropy'):.4f}')
#gini impurity
# pos_fraction = np.linspace(0.00, 1.00, 1000)
# gini = 1 - pos_fraction**2 - (1-pos_fraction)**2
# plt.plot(pos_fraction, gini)
# plt.ylim(0, 1)
# plt.xlabel('Udzial klasy dodatniej')
# plt.ylabel('Zanieczyszczenie Giniego')
# plt.show()
#information gain
# pos_fraction = np.linspace(0.00, 1.00, 1000)
# ent = - (pos_fraction * np.log2(pos_fraction) + (1 - pos_fraction) * np.log2(1 - pos_fraction))
# plt.plot(pos_fraction, ent)
# plt.ylim(0, 1)
# plt.xlabel('Udzial klasy dodatniej')
# plt.ylabel('Entropia')
# plt.show()