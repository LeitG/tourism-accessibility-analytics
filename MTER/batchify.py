import torch
import math
import random
import numpy as np

from config import Config
import pickle

class NegSamplingBatchify:
    r"""
    The function of negative sampling is provided for the label prediction task, and it is only used in the
    training phase.
    """

    def __init__(self, config, shuffle=False):

        user, item, rating, reason_pos, reason_neg = [], [], [], [], []
        with open('../reviews10000.pickle', 'rb') as f:
            data = pickle.load(f)

        res = []
        for line in data:
            if 'sentence' in line:
                quadruples = line['sentence']
                push = {}
                push['user_id'] = line['user']
                push['item_id'] = line['item']
                push['rating'] = line['rating']
                reason = []
                for q in quadruples:
                    if q[3] != -1:
                        reason.append(q[0])
                reason = list(set(reason))
                push['reason_tag'] = reason
                if reason != []:
                    res.append(push)
        self.user_num = len(set(i['user_id'] for i in res ))
        self.item_num = len(set(i['item_id'] for i in res ))
        self.tag_num = len(set([tag for l in res for tag in l['reason_tag']]))
        self.max_rating = max(set(i['rating'] for i in res ))
        self.min_rating = min(set(i['rating'] for i in res ))
        count = 0
        uids = [i['user_id'] for i in res]
        self.user_map = {}
        for uid in uids:
            if uid not in self.user_map:
                self.user_map[uid] = count
                count += 1
        count = 0
        self.tag_map = {}
        tagls = [i['reason_tag'] for i in res]
        for tags in tagls:
            for tag in tags:
                if tag not in self.tag_map:
                    self.tag_map[tag] = count
                    count += 1
        
        count = 0
        items = [i['item_id'] for i in res]
        self.item_map = {}
        for item_ in items:
            if item_ not in self.item_map:
                self.item_map[item_] = count
                count += 1
        for i in range(len(res)):
            res[i]['user_id'] = self.user_map[res[i]['user_id']]
            res[i]['item_id'] = self.item_map[res[i]['item_id']]
            for j in range(len(res[i]['reason_tag'])):
                res[i]['reason_tag'][j] = self.tag_map[res[i]['reason_tag'][j]]
        
        for x in res:
            pos_reason_list = x['reason_tag']


            a_pos_reason_tag = random.choice(pos_reason_list)

            for _ in range(config['neg_sample_num']):
                user.append(x['user_id'])
                item.append(x['item_id'])
                rating.append(x['rating'])
                reason_pos.append(a_pos_reason_tag)

                neg_ra = np.random.randint(self.tag_num)
                while neg_ra in pos_reason_list:
                    neg_ra = np.random.randint(self.tag_num)
                reason_neg.append(neg_ra)


        print(user)
        self.user = torch.tensor(user, dtype=torch.int64)
        self.item = torch.tensor(item, dtype=torch.int64)
        self.rating = torch.tensor(rating, dtype=torch.float)
        self.reason_pos = torch.tensor(reason_pos, dtype=torch.int64)
        self.reason_neg = torch.tensor(reason_neg, dtype=torch.int64)

        self.shuffle = shuffle
        self.batch_size = config['batch_size']
        self.sample_num = len(user)
        self.index_list = list(range(self.sample_num))
        if self.shuffle:
            random.shuffle(self.index_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]
        item = self.item[index]
        rating = self.rating[index]
        reason_pos = self.reason_pos[index]
        reason_neg = self.reason_neg[index]

        return user, item, rating, reason_pos, reason_neg

    def neg_tag_sampling(self):
        return
