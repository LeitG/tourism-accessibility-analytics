import numpy as np
import pickle
from emf_np import EMF_NP
import matplotlib.pyplot as plt


user2idx, user_num = {}, 0
item2idx, item_num = {}, 0
feature2idx, feature_num = {}, 0

with open('./reviews_train.pickle', 'rb') as f:
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

with open('./reviews_test.pickle', 'rb') as f:
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


X = np.zeros((user_num, feature_num), dtype=float)
Y = np.zeros((item_num, feature_num), dtype=float)
A = np.zeros((user_num, item_num), dtype=float)
N = 5

with open('./reviews_train.pickle', 'rb') as f:
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


test_X = np.zeros((user_num, feature_num), dtype=float)
test_Y = np.zeros((item_num, feature_num), dtype=float)

with open('./reviews_test.pickle', 'rb') as f:
    data = pickle.load(f)
    for line in data:
        if 'sentence' in line:
            quadruples = line['sentence']
            cur_user = line['user']
            cur_item = line['item']

            for quadruple in quadruples:
                # if quadruple[0] in feature2idx and cur_user in user2idx and cur_item in item2idx:
                test_X[user2idx[cur_user], feature2idx[quadruple[0]]] += abs(int(quadruple[3]))
                test_Y[item2idx[cur_item], feature2idx[quadruple[0]]] += int(quadruple[3])

for i in range(user_num):
    for j in range(feature_num):
        if test_X[i, j] != 0:
            test_X[i, j] = 1 + (N - 1) * (2 / (1 + np.exp(-test_X[i, j])) - 1)

for i in range(item_num):
    for j in range(feature_num):
        if test_Y[i, j] != 0:
            test_Y[i, j] = 1 + (N - 1) / (1 + np.exp(-test_Y[i, j]))


test_rmse = []
for r in range(0, 101, 10):
    print('r=', r)
    emf_np = EMF_NP(p=feature_num, r=r, Y=Y, T=3000)
    emf_np.fit(A, X)
    test_rmse.append(emf_np.score(test_X, test_Y))
    print(test_rmse[-1])
    print()

plt.plot(range(0, 101, 10), test_rmse, marker='o')
plt.xlabel('Number of Explicit Factors r', fontsize=18)
plt.ylabel('RMSE', fontsize=18)
# plt.ylim(0.1, 0.4)
plt.show()
