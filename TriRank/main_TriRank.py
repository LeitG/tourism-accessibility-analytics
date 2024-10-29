import numpy as np
import pickle
from trirank import TriRank
import matplotlib.pyplot as plt


user2idx, user_num = {}, 0
item2idx, item_num = {}, 0
feature2idx, feature_num = {}, 0

with open('./reviews10000.pickle', 'rb') as f:
    data = pickle.load(f)
    for line in data:
        if 'sentence' in line:
            quadruples = line['sentence']
            cur_user = line['user']
            cur_item = line['item']
            if cur_user not in user2idx:
                user2idx[cur_user] = user_num
                user_num += 1
            if cur_item not in item2idx:
                item2idx[cur_item] = item_num
                item_num += 1
            for quadruple in quadruples:
                if quadruple[0] not in feature2idx:
                    feature2idx[quadruple[0]] = feature_num
                    feature_num += 1


A = np.zeros((user_num, item_num), dtype=float)
X = np.zeros((user_num, feature_num), dtype=float)
Y = np.zeros((item_num, feature_num), dtype=float)
N = 5

with open('./reviews10000.pickle', 'rb') as f:
    data = pickle.load(f)
    for line in data:
        if 'sentence' in line:
            quadruples = line['sentence']
            cur_user = line['user']
            cur_item = line['item']

            for quadruple in quadruples:
                X[user2idx[cur_user], feature2idx[quadruple[0]]] += abs(int(quadruple[3]))
                Y[item2idx[cur_item], feature2idx[quadruple[0]]] += int(quadruple[3])
                A[user2idx[cur_user], item2idx[cur_item]] = line['rating']

for i in range(user_num):
    for j in range(feature_num):
        if X[i, j] != 0:
            X[i, j] = 1 + (N - 1) * (2 / (1 + np.exp(-X[i, j])) - 1)

for i in range(item_num):
    for j in range(feature_num):
        if Y[i, j] != 0:
            Y[i, j] = 1 + (N - 1) / (1 + np.exp(-Y[i, j]))


trirank = TriRank(
    R=A, X=Y, Y=X,
    verbose=True,
    seed=123,
)
print(trirank._online_recommendation(0))
