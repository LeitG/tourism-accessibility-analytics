import os
import torch
import torch.optim as optim

from metrics.metrics import root_mean_square_error, mean_absolute_error, evaluate_precision_recall_f1, evaluate_ndcg

class Trainer(object):

    def __init__(self, config, model, train_data):
        self.config = config
        self.model = model
        self.train_data = train_data

        self.model_name = config['model']
        self.dataset = config['dataset']
        self.epochs = config['epochs']

        self.device = config['device']
        self.batch_size = config['batch_size']
        self.learner = config['learner']
        self.learning_rate = config['lr']
        self.weight_decay = config['weight_decay']
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.95)  # gamma: lr_decay
        self.rating_weight = config['rating_weight']
        self.reason_weight = config['reason_weight']
        self.video_weight = config['video_weight']
        self.interest_weight = config['interest_weight']
        self.l2_weight = config['l2_weight']
        self.top_k = config['top_k']
        self.max_rating = config['max_rating']
        self.min_rating = config['min_rating']

        self.endure_times = config['endure_times']
        self.checkpoint = config['checkpoint']

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def train(self, data):  # train mse+bce+l2
        self.model.train()
        loss_sum = 0.
        Rating_loss = 0.
        Tag_loss = 0.
        L2_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, reason_tag, reason_label, video_tag, video_label, interest_tag, interest_label = data.next_batch()
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            reason_tag = reason_tag.to(self.device)
            video_tag = video_tag.to(self.device)
            interest_tag = interest_tag.to(self.device)
            reason_label = reason_label.to(self.device)
            video_label = video_label.to(self.device)
            interest_label = interest_label.to(self.device)

            self.optimizer.zero_grad()
            rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
            reason_loss = self.model.calculate_reason_loss(user, item, reason_tag, reason_label) * self.reason_weight
            video_loss = self.model.calculate_video_loss(user, item, video_tag, video_label) * self.video_weight
            interest_loss = self.model.calculate_interest_loss(user, item, interest_tag,
                                                               interest_label) * self.interest_weight
            l2_loss = self.model.calculate_l2_loss() * self.l2_weight
            loss = rating_loss + reason_loss + video_loss + interest_loss + l2_loss
            loss.backward()
            self.optimizer.step()
            Rating_loss += self.batch_size * rating_loss.item()
            Tag_loss += self.batch_size * (reason_loss.item() + video_loss.item() + interest_loss.item())
            L2_loss += self.batch_size * l2_loss.item()
            loss_sum += self.batch_size * loss.item()
            total_sample += self.batch_size

            if data.step == data.total_step:
                break
        return Rating_loss / total_sample, Tag_loss / total_sample, L2_loss / total_sample, loss_sum / total_sample

    def valid(self, data):  # valid
        self.model.eval()
        loss_sum = 0.
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, rating, reason_tag, reason_label, video_tag, video_label, interest_tag, interest_label = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                reason_tag = reason_tag.to(self.device)
                video_tag = video_tag.to(self.device)
                interest_tag = interest_tag.to(self.device)
                reason_label = reason_label.to(self.device)
                video_label = video_label.to(self.device)
                interest_label = interest_label.to(self.device)

                rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
                reason_loss = self.model.calculate_reason_loss(user, item, reason_tag,
                                                               reason_label) * self.reason_weight
                video_loss = self.model.calculate_video_loss(user, item, video_tag, video_label) * self.video_weight
                interest_loss = self.model.calculate_interest_loss(user, item, interest_tag,
                                                                   interest_label) * self.interest_weight
                l2_loss = self.model.calculate_l2_loss() * self.l2_weight
                loss = rating_loss + reason_loss + video_loss + interest_loss + l2_loss

                loss_sum += self.batch_size * loss.item()
                total_sample += self.batch_size

                if data.step == data.total_step:
                    break
        return loss_sum / total_sample

    def evaluate(self, model, data):  # test
        model.eval()
        rating_predict = []
        reason_predict = []
        video_predict = []
        interest_predict = []
        with torch.no_grad():
            while True:
                user, item, candi_reason_tag, candi_video_tag, candi_interest_tag = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                reason_candidate_tag = candi_reason_tag.to(self.device)
                video_candidate_tag = candi_video_tag.to(self.device)
                interest_candidate_tag = candi_interest_tag.to(self.device)

                rating_p = model.predict_rating(user, item)  # (batch_size,)
                rating_predict.extend(rating_p.tolist())

                reason_p = model.predict_reason_score(user, item, reason_candidate_tag)  # (batch_size, candidate_num)
                _, reason_p_topk = torch.topk(reason_p, dim=-1, k=self.top_k, largest=True,
                                              sorted=True)  # values & index
                reason_predict.extend(reason_candidate_tag.gather(1, reason_p_topk).tolist())

                video_p = model.predict_video_score(user, item, video_candidate_tag)  # (batch_size,candidate_num)
                _, video_p_topk = torch.topk(video_p, dim=-1, k=self.top_k, largest=True, sorted=True)  # values & index
                video_predict.extend(video_candidate_tag.gather(1, video_p_topk).tolist())

                interest_p = model.predict_interest_score(user, item,
                                                          interest_candidate_tag)  # (batch_size,candidate_num)
                _, interest_p_topk = torch.topk(interest_p, dim=-1, k=self.top_k, largest=True,
                                                sorted=True)  # values & index
                interest_predict.extend(interest_candidate_tag.gather(1, interest_p_topk).tolist())

                if data.step == data.total_step:
                    break
        # rating
        rating_zip = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
        RMSE = root_mean_square_error(rating_zip, self.max_rating, self.min_rating)
        MAE = mean_absolute_error(rating_zip, self.max_rating, self.min_rating)
        # reason_tag
        reason_p, reason_r, reason_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_reason_tag,
                                                                     reason_predict)
        reason_ndcg = evaluate_ndcg(self.top_k, data.positive_reason_tag, reason_predict)
        # video_tag
        video_p, video_r, video_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_video_tag, video_predict)
        video_ndcg = evaluate_ndcg(self.top_k, data.positive_video_tag, video_predict)
        # interest_tag
        interest_p, interest_r, interest_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_interest_tag,
                                                                           interest_predict)
        interest_ndcg = evaluate_ndcg(self.top_k, data.positive_interest_tag, interest_predict)

        return RMSE, MAE, \
               reason_p, reason_r, reason_f1, reason_ndcg, \
               video_p, video_r, video_f1, video_ndcg, \
               interest_p, interest_r, interest_f1, interest_ndcg

    def train_loop(self):
        best_val_loss = float('inf')
        best_epoch = 0
        endure_count = 0
        model_path = ''
        for epoch in range(1, self.epochs + 1):
            print('epoch {}'.format(epoch))
            train_r_loss, train_t_loss, train_l_loss, train_sum_loss = self.train(self.train_data)
            print(
                'rating loss {:4.4f} | tag loss {:4.4f} | l2 loss {:4.4f} |total loss {:4.4f} on train'.format(
                    train_r_loss, train_t_loss, train_l_loss, train_sum_loss))
            # val_loss = self.valid(self.val_data)
            val_loss = 0.0
            print('total loss {:4.4f} on validation'.format(val_loss))

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_val_loss:
                saved_model_file = '{}-{}.pt'.format(self.model_name, self.dataset)
                model_path = os.path.join(self.checkpoint, saved_model_file)
                with open(model_path, 'wb') as f:
                    torch.save(self.model, f)
                print('Save the best model' + model_path)
                best_val_loss = val_loss
                best_epoch = epoch
            else:
                endure_count += 1
                print('Endured {} time(s)'.format(endure_count))
                if endure_count == self.endure_times:
                    print('Cannot endure it anymore | Exiting from early stop')
                    break
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                self.scheduler.step()
                print('Learning rate set to {:2.8f}'.format(self.scheduler.get_last_lr()[0]))

        return model_path, best_epoch


class MTERTrainer(Trainer):
    def __init__(self, config, model, train_data):
        super(MTERTrainer, self).__init__(config, model, train_data)
        self.non_neg_weight = config['non_neg_weight']

    def train(self, data):
        self.model.train()
        loss_sum = 0.
        Rating_loss = 0.    
        Tag_loss = 0.
        L2_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, reason_pos, reason_neg = data.next_batch()

            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            reason_pos = reason_pos.to(self.device)
            reason_neg = reason_neg.to(self.device)

            self.optimizer.zero_grad()
            rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
            reason_loss = self.model.calculate_reason_loss(user, item, reason_pos, reason_neg) * self.reason_weight
            non_neg_loss = self.model.calculate_non_negative_reg() * self.non_neg_weight
            loss = rating_loss + reason_loss + non_neg_loss

            loss.backward()
            self.optimizer.step()
            Rating_loss += self.batch_size * rating_loss.item()
            Tag_loss += self.batch_size * (reason_loss.item())
            loss_sum += self.batch_size * loss.item()
            total_sample += self.batch_size

            if data.step == data.total_step:
                break
        return Rating_loss / total_sample, Tag_loss / total_sample, L2_loss / total_sample, loss_sum / total_sample

    def valid(self, data):  # valid
        self.model.eval()
        loss_sum = 0.
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, rating, reason_pos, reason_neg, video_pos, video_neg, interest_pos, interest_neg = data.next_batch()

                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                reason_pos = reason_pos.to(self.device)
                reason_neg = reason_neg.to(self.device)
                video_pos = video_pos.to(self.device)
                video_neg = video_neg.to(self.device)
                interest_pos = interest_pos.to(self.device)
                interest_neg = interest_neg.to(self.device)

                rating_loss = self.model.calculate_rating_loss(user, item, rating) * self.rating_weight
                reason_loss = self.model.calculate_reason_loss(user, item, reason_pos, reason_neg) * self.reason_weight
                video_loss = self.model.calculate_video_loss(user, item, video_pos, video_neg) * self.video_weight
                interest_loss = self.model.calculate_interest_loss(user, item, interest_pos,
                                                                   interest_neg) * self.interest_weight
                non_neg_loss = self.model.calculate_non_negative_reg() * self.non_neg_weight
                loss = rating_loss + reason_loss + video_loss + interest_loss + non_neg_loss

                loss_sum += self.batch_size * loss.item()
                total_sample += self.batch_size

                if data.step == data.total_step:
                    break
        return loss_sum / total_sample

    def evaluate(self, model, data):  # test
        model.eval()
        rating_predict = []
        reason_predict = []
        video_predict = []
        interest_predict = []
        with torch.no_grad():
            while True:
                user, item, candi_reason_tag, candi_video_tag, candi_interest_tag = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                reason_candidate_tag = candi_reason_tag.to(self.device)
                video_candidate_tag = candi_video_tag.to(self.device)
                interest_candidate_tag = candi_interest_tag.to(self.device)

                rating_p = model.predict_rating(user, item)  # (batch_size,)
                rating_predict.extend(rating_p.tolist())

                reason_p = model.rank_reason_tags(user, item, reason_candidate_tag)  # (batch_size, tag_num)
                _, reason_p_topk = torch.topk(reason_p, dim=-1, k=self.top_k, largest=True,
                                              sorted=True)  # values & index
                reason_predict.extend(reason_candidate_tag.gather(1, reason_p_topk).tolist())

                video_p = model.rank_video_tags(user, item, video_candidate_tag)  # (batch_size,tag_num)
                _, video_p_topk = torch.topk(video_p, dim=-1, k=self.top_k, largest=True, sorted=True)  # values & index
                video_predict.extend(video_candidate_tag.gather(1, video_p_topk).tolist())

                interest_p = model.rank_interest_tags(user, item, interest_candidate_tag)  # (batch_size,tag_num)
                _, interest_p_topk = torch.topk(interest_p, dim=-1, k=self.top_k, largest=True,
                                                sorted=True)  # values & index
                interest_predict.extend(interest_candidate_tag.gather(1, interest_p_topk).tolist())

                if data.step == data.total_step:
                    break
        # rating
        rating_zip = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
        RMSE = root_mean_square_error(rating_zip, self.max_rating, self.min_rating)
        MAE = mean_absolute_error(rating_zip, self.max_rating, self.min_rating)
        # reason_tag
        reason_p, reason_r, reason_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_reason_tag,
                                                                     reason_predict)
        reason_ndcg = evaluate_ndcg(self.top_k, data.positive_reason_tag, reason_predict)
        # video_tag
        video_p, video_r, video_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_video_tag, video_predict)
        video_ndcg = evaluate_ndcg(self.top_k, data.positive_video_tag, video_predict)
        # interest_tag
        interest_p, interest_r, interest_f1 = evaluate_precision_recall_f1(self.top_k, data.positive_interest_tag,
                                                                           interest_predict)
        interest_ndcg = evaluate_ndcg(self.top_k, data.positive_interest_tag, interest_predict)

        return RMSE, MAE, \
               reason_p, reason_r, reason_f1, reason_ndcg, \
               video_p, video_r, video_f1, video_ndcg, \
               interest_p, interest_r, interest_f1, interest_ndcg
