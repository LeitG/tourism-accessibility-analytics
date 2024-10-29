import torch
import numpy as np
import os
import tqdm
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from argument_amazon import arg_parse_train_base, arg_parser_preprocessing
from data_loaders import UserItemInterDataset
from models import BaseRecModel
from evaluate_functions import compute_ndcg


def train_base_recommendation(train_args, pre_processing_args):
    if train_args.gpu:
        device = torch.device('cuda:%s' % train_args.cuda)
    else:
        device = 'cpu'


    Path(pre_processing_args.save_path).mkdir(parents=True, exist_ok=True)
    # with open(os.path.join(pre_processing_args.save_path, pre_processing_args.dataset + "_dataset_obj.pickle"), 'wb') as outp:
    #     pickle.dump(rec_dataset, outp, pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.dirname(os.path.realpath(__file__)) + '/datasets/my/train_data.pickle', 'rb') as f:
        training_data = pickle.load(f)

    with open(os.path.dirname(os.path.realpath(__file__)) + '/datasets/my/user_feature_matrix.pickle', 'rb') as f:
        user_feature_matrix = pickle.load(f).astype(np.float32)

    with open(os.path.dirname(os.path.realpath(__file__)) + '/datasets/my/item_feature_matrix.pickle', 'rb') as f:
        item_feature_matrix = pickle.load(f).astype(np.float32)
    with open(os.path.dirname(os.path.realpath(__file__)) + '/datasets/my/test_data.pickle', 'rb') as f:
        test_data = pickle.load(f)
    train_loader = DataLoader(dataset=UserItemInterDataset(training_data, 
                                user_feature_matrix, 
                                item_feature_matrix),
                          batch_size=train_args.batch_size,
                          shuffle=True)

    model = BaseRecModel(item_feature_matrix.shape[1]).to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=train_args.lr, weight_decay=train_args.weight_decay)

    out_path = os.path.join("./logs", train_args.dataset + "_logs")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    ndcg = compute_ndcg(test_data, 
            user_feature_matrix, 
            item_feature_matrix, 
            train_args.rec_k, 
            model, 
            device)
    print('init ndcg:', ndcg)
    for epoch in tqdm.trange(train_args.epoch):
        model.train()
        optimizer.zero_grad()
        losses = []
        for user_behaviour_feature, item_aspect_feature, label in train_loader:
            user_behaviour_feature = user_behaviour_feature.to(device)
            item_aspect_feature = item_aspect_feature.to(device)
            label = label.float().to(device)
            out = model(user_behaviour_feature, item_aspect_feature).squeeze()
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.to('cpu').detach().numpy())
            ave_train = np.mean(np.array(losses))
        print('epoch %d: ' % epoch, 'training loss: ', ave_train)
        # compute necg
        if epoch % 100 == 0:
            ndcg = compute_ndcg(test_data, 
            user_feature_matrix, 
            item_feature_matrix, 
            train_args.rec_k, 
            model, 
            device)
            print('epoch %d: ' % epoch, 'training loss: ', ave_train, 'NDCG: ', ndcg)
    torch.save(model.state_dict(), os.path.join(out_path, "model.model"))
    return 0


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    t_args = arg_parse_train_base()  # training arguments
    p_args = arg_parser_preprocessing()  # pre processing arguments
    if t_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = t_args.cuda
        print("Using CUDA", t_args.cuda)
    else:
        print("Using CPU")
    train_base_recommendation(t_args, p_args)