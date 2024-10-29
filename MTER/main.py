import os
import torch
import argparse
import random
import datetime
import numpy as np
import importlib

from config import Config
from dataloader import TagDataLoader
from batchify import NegSamplingBatchify
from model import MTER
from trainer import MTERTrainer

def set_seed(seed):
    r"""
    set seed for random sampling.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


config = Config(config_file_list=['MTER.yaml']).final_config_dict
print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for param in config:
    print('{:40} {}'.format(param, config[param]))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)
if not os.path.exists(config['checkpoint']):
    os.makedirs(config['checkpoint'])

# Set the random seed manually for reproducibility.
set_seed(config['seed'])




# validset_size = corpus.valid_size
# testset_size = corpus.test_size

train_data = NegSamplingBatchify(config, shuffle=True)
tag_num = train_data.tag_num
user_num = train_data.user_num
item_num = train_data.item_num
# trainset_size = train_data.train_size
# val_data = NegSamplingBatchify(corpus.validset, config, tag_num)
# test_data = NegSamplingBatchify(corpus.testset, config, tag_num)

###############################################################################
# Update Config
###############################################################################

config['user_num'] = user_num
config['item_num'] = item_num
config['tag_num'] = tag_num
config['max_rating'] = train_data.max_rating
config['min_rating'] = train_data.min_rating
config['device'] = 'cpu'

###############################################################################
# Build the model
###############################################################################
print(config)
model = MTER(config).to('cpu')
trainer = MTERTrainer(config, model, train_data)
###############################################################################
# Loop over epochs
###############################################################################

model_path, best_epoch = trainer.train_loop()

# Load the best saved model.
# with open(model_path, 'rb') as f:
#     model = torch.load(f).to(device)
# print(now_time() + 'Load the best model' + model_path)

# # Run on test data.
# rmse, mse, \
# reason_p, reason_r, reason_f1, reason_ndcg, \
# video_p, video_r, video_f1, video_ndcg, \
# interest_p, interest_r, interest_f1, interest_ndcg = trainer.evaluate(model)
# print('=' * 89)
# # Results
# print('Best model in epoch {}'.format(best_epoch))
# print('Best results: RMSE {:7.4f} | MAE {:7.4f}'.format(rmse, mse))
# print('Best test: RMSE {:7.4f} | MAE {:7.4f}'.format(rmse, mse))
# print('Best test: reason_tag   @{} precision{:7.4f} | recall {:7.4f} | f1 {:7.5f} | ndcg {:7.5f}'
#       .format(config['top_k'], reason_p, reason_r, reason_f1, reason_ndcg))
# print('Best test: video_tag    @{} precision{:7.4f} | recall {:7.4f} | f1 {:7.5f} | ndcg {:7.5f}'
#       .format(config['top_k'], video_p, video_r, video_f1, video_ndcg))
# print('Best test: interest_tag @{} precision{:7.4f} | recall {:7.4f} | f1 {:7.5f} | ndcg {:7.5f}'
#       .format(config['top_k'], interest_p, interest_r, interest_f1, interest_ndcg))
