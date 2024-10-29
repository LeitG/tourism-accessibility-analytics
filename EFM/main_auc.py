import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

from emf_np import EMF_NP

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

test_data = []
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

            test_data.append([user2idx[line['user']], item2idx[line['item']], line['rating']])
test_data = np.array(test_data)

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
                # rate = line['rating']
                # if rate > 3.5:
                #     A[user2idx[cur_user], item2idx[cur_item]] = 1
                # else:
                #     A[user2idx[cur_user], item2idx[cur_item]] = 0
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


# T=5000
emf_np = EMF_NP(p=feature_num, r=20, Y=Y, T=5000)
emf_np.fit(A, X)
R = emf_np.get_all_rating()

# Normalized Discounted Cumulative Gain(NDCG) and
# Area Under the ROC Curve (AUC) to help evaluate the performance.

predicted_ratings = np.array([R[user, item] for user, item, _ in test_data])
true_ratings = np.array([true_rating for _, _, true_rating in test_data])

top1_predicted = np.argsort(-predicted_ratings)[:1]
top1_true = np.argsort(-true_ratings)[:1]


def ndcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        dcg = r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        max_dcg = np.sort(r)[::-1]
        max_dcg = max_dcg[0] + np.sum(max_dcg[1:] / np.log2(np.arange(2, max_dcg.size + 1)))
        return dcg / max_dcg
    return 0.


ndcg_score = ndcg_at_k(true_ratings[top1_predicted], 1)

threshold = 3.5
binary_true_ratings = (true_ratings > threshold).astype(int)
binary_predicted_ratings = (predicted_ratings > threshold).astype(int)

auc_score = roc_auc_score(binary_true_ratings, predicted_ratings)

print(f"NDCG: {ndcg_score}")
print(f"AUC: {auc_score}")
