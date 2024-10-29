import pandas as pd
import os
import torch
import heapq

class TagDataLoader:

    def __init__(self, data_path, video_path, train_path, valid_path, test_path):
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.rating_scale = 5
        self.item_set = set()
        self.user_set = set()
        self.tag_num = 0
        self.interaction_num = 0
        self.initialize(data_path, video_path)
        self.user_num = len(self.user_set)
        self.item_num = len(self.item_set)
        self.trainData, self.trainset, self.validset, self.testset = self.load_data(train_path, valid_path, test_path)

        self.train_size = len(self.trainset)
        self.valid_size = len(self.validset)
        self.test_size = len(self.testset)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pd.read_csv(data_path, header=0,
                              names=['user_id', 'video_id', 'rating', 'reason_tag'],
                              sep='\t')
        reviews = reviews.to_dict('records')
        tag_list = [tag for reason_row in reviews['reason_tag'] for tag in eval(reason_row)]
        self.tag_num = len(set(tag_list)) + 7
        for review in reviews:
            self.user_set.add(review['user_id'])
            self.item_set.add(review['video_id'])
            self.interaction_num += 1
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, train_path, valid_path, test_path):
        train_data = pd.read_csv(train_path, header=0,
                                names=['user_id', 'video_id', 'rating', 'reason_tag'],
                                sep='\t')
        valid_data = pd.read_csv(valid_path, header=0,
                              names=['user_id', 'video_id', 'rating', 'reason_tag'],
                              sep='\t')
        test_data = pd.read_csv(test_path, header=0,
                              names=['user_id', 'video_id', 'rating', 'reason_tag'],
                              sep='\t')

        trainset = train_data.to_dict('records')
        validset = valid_data.to_dict('records')
        testset = test_data.to_dict('records')  # list [{'user_id':1, 'video_id':1, 'reason_tag':'[5, 2, 1,..]'}, {...}]

        return train_data, trainset, validset, testset

