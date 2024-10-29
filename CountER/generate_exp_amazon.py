import torch
import pickle
import os
from pathlib import Path
import numpy as np
from argument_amazon import arg_parse_exp_optimize
from models import BaseRecModel, ExpOptimizationModel

class REC_DATASET:
    def __init__(self) -> None:
        with open(os.path.dirname(os.path.realpath(__file__)) + '/datasets/my/user_feature_matrix.pickle', 'rb') as f:
            self.user_feature_matrix = pickle.load(f).astype(np.float32)

        with open(os.path.dirname(os.path.realpath(__file__)) + '/datasets/my/item_feature_matrix.pickle', 'rb') as f:
            self.item_feature_matrix = pickle.load(f).astype(np.float32)
        with open(os.path.dirname(os.path.realpath(__file__)) + '/datasets/my/test_data.pickle', 'rb') as f:
            self.test_data = pickle.load(f)
        with open(os.path.dirname(os.path.realpath(__file__)) + '/datasets/my/sentiment_data.pickle', 'rb') as f:
            self.sentiment_data = pickle.load(f)
        self.feature_num = self.item_feature_matrix.shape[1]

def generate_explanation(exp_args):
    if exp_args.gpu:
        device = torch.device('cuda:%s' % exp_args.cuda)
    else:
        device = 'cpu'
    # import dataset
    # with open(os.path.join(exp_args.data_obj_path, exp_args.dataset + "_dataset_obj.pickle"), 'rb') as inp:
    #     rec_dataset = pickle.load(inp)
    rec_dataset = REC_DATASET()
    base_model = BaseRecModel(rec_dataset.feature_num).to(device)
    base_model.load_state_dict(torch.load(os.path.join(exp_args.base_model_path, exp_args.dataset+"_logs", "model.model")))
    base_model.eval()
    #  fix the rec model
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Create optimization model
    opt_model = ExpOptimizationModel(
        base_model=base_model,
        rec_dataset=rec_dataset,
        device = device,
        exp_args=exp_args,
    )

    opt_model.generate_explanation()
    opt_model.user_side_evaluation()
    opt_model.model_side_evaluation()
    Path(exp_args.save_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(exp_args.save_path, exp_args.dataset + "_explanation_obj.pickle"), 'wb') as outp:
        pickle.dump(opt_model, outp, pickle.HIGHEST_PROTOCOL)
    return True


if __name__ == "__main__":
    exp_args = arg_parse_exp_optimize()
    generate_explanation(exp_args)