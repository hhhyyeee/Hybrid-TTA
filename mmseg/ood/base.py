
import math
import random

import torch
import numpy as np

from mmseg.utils import get_root_logger

from tools.get_param_count import count_parameters


class FreezeController(object):
    def __init__(self, cfg, show_dir=None, logger=None):
        self.cfg = cfg
        self.mode = cfg.get("ood_mode", "manual")
        assert self.mode in ['manual', 'loss_seg', 'entropy', 'manual-entropy', 'random_test',
                             'loss_seg_with_buffer', 'loss_seg_with_wma', 'contextual_thompson_sampling']

        # logger
        self.freeze_logger = get_root_logger(log_file=f"{show_dir}/freeze.log", log_level="INFO") if show_dir else None
        self.logger = logger if logger else None

        self.freeze_intermediate = cfg.get('freeze_intermediate', [])
        self.except_keywords = cfg.get('except_keywords', [])
        self.freeze_cfg = dict(modules=self.freeze_intermediate, except_keywords=self.except_keywords)

        if self.mode == 'manual':
            self.freeze_epoch = cfg.get('freeze_epoch', 10)
            self.freeze_iter = cfg.get('freeze_iter', 400)

        elif self.mode == 'loss_seg':
            self.threshold = cfg.get('loss_seg_threshold', 0.1)
            self.loss_seg_alpha = cfg.get('loss_seg_alpha', 0.99)

        elif self.mode == 'entropy':
            self.threshold = cfg.get('entropy_threshold', 0.2)
            self.entropy_alpha = cfg.get('entropy_alpha', 0.99)

        elif self.mode == 'manual-entropy':
            self.freeze_epoch = cfg.get('freeze_epoch', 10)
            self.freeze_iter = cfg.get('freeze_iter', 400)
            self.threshold = cfg.get('entropy_threshold', 0.2)
            self.entropy_alpha = cfg.get('entropy_alpha', 0.99)

        elif self.mode == 'random_test':
            random.seed(42)
            self.random_test_prob = cfg.get('random_test_prob', 0.9)

        elif self.mode == 'loss_seg_with_buffer':
            self.threshold = 0.2
            self.ema_loss = 0.2
            self.loss_seg_alpha = 0.99
            self.threshold_alpha = 0.999
            self.buffer = 0.2

        elif self.mode == 'loss_seg_with_wma':
            self.threshold = cfg.get('loss_seg_threshold', 0.1)
            self.loss_seg_alpha = cfg.get('loss_seg_alpha', 0.99)
            self.window_size = cfg.get('wma_window_size', 30)
            self.perc = cfg.get('wma_perc', 0.2)
            self.window = []
            self.wma_loss = 0.0
            self.et_flag = False

        elif self.mode == 'contextual_thompson_sampling':
            self.window_size = cfg.get('wma_window_size', 30)
            self.window = []
            self.wma_loss = 0.

            self.actions = ['efficient_training', 'full_training']
            self.reward = None
            self.num_actions = len(self.actions)
            self.action_alpha = np.ones(self.num_actions)
            self.action_beta = np.ones(self.num_actions)
            self.state_dim = 3
            self.mu = np.zeros((self.num_actions, self.state_dim))
            self.cov = np.array([np.eye(self.state_dim) for _ in range(self.num_actions)])
            self.beta = 1.

        self.frozen_flag = False

    def __call__(self, *args, **kwargs):
        if self.mode == 'manual':
            return self.manual(*args, **kwargs)
        elif self.mode == 'loss_seg':
            return self.loss_seg(*args, **kwargs)
        elif self.mode == 'entropy':
            return self.entropy(*args, **kwargs)
        elif self.mode == 'manual-entropy':
            return self.manual_entropy(*args, **kwargs)
        elif self.mode == 'random_test':
            return self.random_test(*args, **kwargs)
        elif self.mode == 'loss_seg_with_buffer':
            return self.loss_seg_with_buffer(*args, **kwargs)
        elif self.mode == 'loss_seg_with_wma':
            return self.loss_seg_with_wma(*args, **kwargs)
        elif self.mode == 'contextual_thompson_sampling':
            return self.contextual_thompson_sampling(*args, **kwargs)
        else:
            pass

    def manual(self, model, epoch_index, iter_index=0, *args, **kwargs):
        if self.frozen_flag:
            # in manual mode, if already frozen, it doesn't need to be reconsidered within an epoch
            # and `reset` is called at the beginning of every epoch
            return

        if epoch_index >= self.freeze_epoch:
            model.module.freeze_components(**self.freeze_cfg)
            if self.logger:
                self.logger.info(f"epoch {epoch_index} # of trainable params: {count_parameters(model, logger=None)}")
            if self.freeze_logger:
                self.freeze_logger.info(f"epoch {epoch_index} # of trainable params: {count_parameters(model, logger=None)}")

            self.frozen_flag = True
            return

        if iter_index >= self.freeze_iter:
            if self.logger:
                self.logger.info(f"epoch {epoch_index}, iter {iter_index} >= freeze_iter {self.freeze_iter}")
            if self.freeze_logger:
                self.freeze_logger.info(f"epoch {epoch_index}, iter {iter_index} >= freeze_iter {self.freeze_iter}")
            model.module.freeze_components(**self.freeze_cfg)
            self.frozen_flag = True
            return

        return

    def loss_seg(self, model, epoch_index, iter_index, *args, **kwargs):
        if 'loss' not in kwargs: return
        loss_seg = kwargs['loss'][1]['decode.loss_seg']
        self.freeze_logger.info(f"[{epoch_index} - {iter_index}] loss_seg - loss_seg {loss_seg}")
        if loss_seg <= self.threshold:
            # if loss_seg_log is less than or equal to threshold, efficient tuning
            if not self.frozen_flag:
                self.freeze_logger.info(f"[{epoch_index} - {iter_index}] loss_seg - loss_seg {loss_seg:15.5f} <= {self.threshold} -> freeze")
                model.module.freeze_components(**self.freeze_cfg)
                self.frozen_flag = True
        else:
            if self.frozen_flag:
                self.freeze_logger.info(f"[{epoch_index} - {iter_index}] loss_seg - loss_seg {loss_seg:15.5f} > {self.threshold} -> unfreeze")
                model.module.unfreeze_components(**self.freeze_cfg)
                self.frozen_flag = False
        self.threshold = self.threshold * self.loss_seg_alpha + loss_seg * (1 - self.loss_seg_alpha)
        return

    def entropy(self, model, epoch_index, iter_index, *args, **kwargs):
        if 'log_vars' not in kwargs: return
        loss = kwargs['loss']
        log_vars = kwargs['log_vars']
        softmax_entropy = kwargs['softmax_entropy']
        if softmax_entropy <= self.threshold:
            if not self.frozen_flag:
                self.freeze_logger.info(f"[{epoch_index} - {iter_index}] entropy - softmax_entropy {softmax_entropy:15.5f} <= {self.threshold} -> freeze")
                model.module.freeze_components(**self.freeze_cfg)
                self.frozen_flag = True
        else:
            if self.frozen_flag:
                self.freeze_logger.info(f"[{epoch_index} - {iter_index}] entropy - softmax_entropy {softmax_entropy:15.5f} > {self.threshold} -> unfreeze")
                model.module.unfreeze_components(**self.freeze_cfg)
                self.frozen_flag = False
        self.threshold = self.threshold * self.entropy_alpha + softmax_entropy * (1 - self.entropy_alpha)
        return

    def manual_entropy(self, model, epoch_index, iter_index, *args, **kwargs):
        if 'log_vars' not in kwargs: return

        if epoch_index < self.freeze_epoch:
            # full tuning mode (warm-up)
            if self.frozen_flag:
                assert False
            softmax_entropy = kwargs['softmax_entropy']
            self.threshold = self.threshold * self.entropy_alpha + softmax_entropy * (1 - self.entropy_alpha)
            return

        else:
            # entropy thresholding
            loss = kwargs['loss']
            log_vars = kwargs['log_vars']
            softmax_entropy = kwargs['softmax_entropy']
            if softmax_entropy <= self.threshold:
                if not self.frozen_flag:
                    self.freeze_logger.info(f"[{epoch_index} - {iter_index}] entropy - softmax_entropy {softmax_entropy:15.5f} <= {self.threshold} -> freeze")
                    model.module.freeze_components(**self.freeze_cfg)
                    self.frozen_flag = True
            else:
                if self.frozen_flag:
                    self.freeze_logger.info(f"[{epoch_index} - {iter_index}] entropy - softmax_entropy {softmax_entropy:15.5f} > {self.threshold} -> unfreeze")
                    model.module.unfreeze_components(**self.freeze_cfg)
                    self.frozen_flag = False

            self.threshold = self.threshold * self.entropy_alpha + softmax_entropy * (1 - self.entropy_alpha)

        return

    def random_test(self, model, epoch_index, *args, **kwargs):
        """
        freeze_intermediate: 검사 대상인 레이어/모듈 등 (여기 포함 안되는 애들은 전부 trainable)
        except_keywords: 학습할 파라미터 키워드
        """
        modules = self.freeze_intermediate
        except_keywords = []
        for name, param in model.named_parameters():
            #검사 대상인 레이어에 포함되어 있지 않은 경우 pass (patch_embed1, decoder, norm 등)
            flag = np.array([(x in name) for x in modules]).sum()
            if flag == 0: continue
            #나머지 애들은 random한 확률로 껐다켰다
            if random.random() < self.random_test_prob:
                except_keywords.append(name)

        freeze_cfg = {
            'modules': modules,
            'except_keywords': except_keywords
        }
        model.module.freeze_components(**freeze_cfg)

    def loss_seg_with_buffer(self, model, epoch_index, iter_index, *args, **kwargs):
        if 'loss' not in kwargs: return
        loss_seg = kwargs['loss'][1]['decode.loss_seg']

        # Seg Loss EMA
        self.ema_loss = self.ema_loss * self.loss_seg_alpha + loss_seg * (1-self.loss_seg_alpha)

        et_flag = abs(self.threshold - self.ema_loss) <= self.buffer

        if et_flag:
            # if loss_seg_log is less than or equal to threshold, efficient tuning
            if not self.frozen_flag:
                self.freeze_logger.info(f"[{epoch_index} - {iter_index}] loss_seg - loss_seg {self.ema_loss:15.5f} <= {self.threshold} -> freeze")
                model.module.freeze_components(**self.freeze_cfg)
                self.frozen_flag = True
        else:
            if self.frozen_flag:
                self.freeze_logger.info(f"[{epoch_index} - {iter_index}] loss_seg - loss_seg {self.ema_loss:15.5f} > {self.threshold} -> unfreeze")
                model.module.unfreeze_components(**self.freeze_cfg)
                self.frozen_flag = False

        self.threshold = self.threshold * self.threshold_alpha + loss_seg * (1 - self.threshold_alpha)
        return

    def loss_seg_with_wma(self, model, epoch_index, iter_index, *args, **kwargs):
        if 'loss' not in kwargs: return
        loss_seg = kwargs['loss'][1]['decode.loss_seg']

        self.window.append(loss_seg)
        if len(self.window) < self.window_size:
            et_flag = False
        else:
            self.wma_loss = np.mean(self.window[-self.window_size:])
            et_flag = self.wma_loss < (1 + self.perc) * self.threshold

        if et_flag:
            # if loss_seg_log is less than or equal to threshold, efficient tuning
            if not self.frozen_flag:
                self.freeze_logger.info(f"[{epoch_index} - {iter_index}] loss_seg - loss_seg {loss_seg:15.5f} <= {self.threshold} -> freeze")
                model.module.freeze_components(**self.freeze_cfg)
                self.frozen_flag = True
        else:
            if self.frozen_flag:
                self.freeze_logger.info(f"[{epoch_index} - {iter_index}] loss_seg - loss_seg {loss_seg:15.5f} > {self.threshold} -> unfreeze")
                model.module.unfreeze_components(**self.freeze_cfg)
                self.frozen_flag = False

        self.threshold = self.threshold * self.loss_seg_alpha + loss_seg * (1 - self.loss_seg_alpha)
        return

    def contextual_thompson_sampling(self, model, epoch_index, iter_index, loss, *args, **kwargs):
        # if 'loss' not in kwargs: return

        loss = loss.get('decode.loss_seg')
        AGM = kwargs.get('total_norm')
        # self.reward = None

        self.window.append(loss)
        if len(self.window) < self.window_size:
            et_flag = False
        else:
            self.wma_loss = np.mean(self.window[-self.window_size:])
            LMA = self.wma_loss
            state = (LMA, loss, AGM)

            action = self.select_action_thompson(state)
            if action == 'efficient_training':
                et_flag = True
                time_cost = 1
            else:
                et_flag = False
                time_cost = 10

            loss_reduction = self.window[-2] - self.window[-1]
            self.reward = loss_reduction / np.sqrt(time_cost + 0.1)

            self.update_posterior(action, state, self.reward)

        if et_flag:
            # if loss_seg_log is less than or equal to threshold, efficient tuning
            if not self.frozen_flag:
                # self.freeze_logger.info(f"[{epoch_index} - {iter_index}] loss_seg - loss_seg {loss_seg:15.5f} <= {self.threshold} -> freeze")
                model.module.freeze_components(**self.freeze_cfg)
                self.frozen_flag = True
        else:
            if self.frozen_flag:
                # self.freeze_logger.info(f"[{epoch_index} - {iter_index}] loss_seg - loss_seg {loss_seg:15.5f} > {self.threshold} -> unfreeze")
                model.module.unfreeze_components(**self.freeze_cfg)
                self.frozen_flag = False

        return

    def select_action_thompson(self, state):
        """ Contextual Thompson Sampling: Predicts reward based on state """
        state = np.array(state).reshape(-1, 1)  # Convert to column vector

        sampled_rewards = []
        for i in range(self.num_actions):
            sampled_weights = np.random.multivariate_normal(self.mu[i], self.cov[i])  # Sample weights
            reward_pred = np.dot(sampled_weights, state.flatten())  # Predict reward
            sampled_rewards.append(reward_pred)

        return self.actions[np.argmax(sampled_rewards)]  # Choose action with highest predicted reward

    def update_posterior(self, action, state, reward):
        """ Bayesian update for action's weight distribution """
        idx = self.actions.index(action)
        state = np.array(state).reshape(-1, 1)  # Column vector

        # Bayesian Linear Regression update
        self.cov[idx] = np.linalg.inv(np.linalg.inv(self.cov[idx]) + self.beta * np.dot(state, state.T))
        self.mu[idx] = np.dot(self.cov[idx], np.dot(np.linalg.inv(self.cov[idx]), self.mu[idx]) + self.beta * reward * state.flatten())


    def reset(self, model, *args, **kwargs):
        model.module.unfreeze_components(**self.freeze_cfg)
        self.frozen_flag = False
