import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.utils import freeze
import mmengine

from .. import builder
from ..builder import SEGMENTORS
from .simmim_encoder_decoder import SimMIMEncoderDecoder

from typing import List
import numpy as np
import scipy
from sklearn.covariance import EmpiricalCovariance
import json
from tqdm import tqdm
from collections import OrderedDict
import math

import random
random.seed(42)
# from tools.our_simmim import random_seed
# random.seed(random_seed)


def get_gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""

    def _gaussian_fn(kernlen, std):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= std
        w = torch.exp(-0.5 * n**2)
        return w

    gkern1d = _gaussian_fn(kernlen, std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()


class HOGLayerC(nn.Module):
    def __init__(self, nbins=9, pool=7, gaussian_window=0):
        super(HOGLayerC, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer("gkern", gkern)

    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx_rgb = F.conv2d(x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w), dtype=torch.float, device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, "h {} gw {}".format(
                    h, self.gaussian_window
                )
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        out = torch.nn.functional.normalize(out, p=2, dim=2)

        return out  # B 3 nbins H W


@SEGMENTORS.register_module()
class MOODEncoderDecoder(SimMIMEncoderDecoder):
    """
    - base task: Semantic Segmentation
    - masked image modeling
    """
    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        mask_cfg=None,
        init_cfg=None,
        mood_cfg=None,
        **cfg
        ):
        super(MOODEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            mask_cfg=mask_cfg,
            init_cfg=init_cfg,
            **cfg
        )
        a=1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hog_recon = False #mask_cfg.get("hog_recon", False)
        if self.hog_recon:
            self.hog = HOGLayerC()

        self.cfg = cfg


    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if kwargs.get('module', None):
            return self.forward_module(img, img_metas, return_loss=return_loss, **kwargs)
        else:
            return super().forward(img, img_metas, return_loss=return_loss, **kwargs)

    def extract_feat(self, img, mask=None, module=None):
        """Extract features from images."""
        if module:
            x = self.backbone.forward_module(img, module=module)
        else:
            x = self.backbone(img, mask=mask)
        if self.with_neck:
            x = self.neck(x)
        return x


    def forward_train_simmim(self, img, img_metas, gt_semantic_seg, seg_weight=None, return_feat=False, **kwargs):

        # masking
        mask = self.mask_generator().to("cuda") #!DEBUG

        # feature extraction w/ mask
        img_latent = self.extract_feat(img, mask=mask)

        a=1
        # _result = {}
        # _result = self.mood_demo(img_latent, img_metas)
        # self.mood_result_list.append(_result)

        losses = dict()
        if return_feat:
            losses['features'] = img_latent

        # reconstruction
        img_rec = self.recon_neck(img_latent)

        if True: #!DEBUG
            a=1
            self.misc.update({
                "img_rec": img_rec.detach().cpu(),
                "img_metas": img_metas})

        if self.hog_recon:
            hog = self.hog(img)
            hog = hog.flatten(1, 2)

        # recon loss
        recon_loss = self.recon_head.loss(img_rec, img, mask)
        losses.update({"recon_loss": recon_loss}) #!DEBUG

        # decode loss w/ mask
        loss_decode = self._decode_head_forward_train(img_latent, img_metas, gt_semantic_seg)

        a=1
        if loss_decode.get('decode.seg_logit', None) is not None:
            seg_logit = loss_decode['decode.seg_logit']
            softmax_entropy = -(seg_logit.softmax(1) * seg_logit.log_softmax(1)).sum(1)
            loss_decode.pop('decode.seg_logit')
            loss_decode.update(dict(softmax_entropy=softmax_entropy))

        losses.update(loss_decode)

        #!DEBUG
        loss_seg = losses['decode.loss_seg']
        # import string
        # import random
        # loss_np = loss_decode['decode.loss_seg'].clone().detach().cpu().numpy()
        # fn = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        # np.save(f'/workspace/CVP-cotta-acdc/notebooks/S00116/losses-200mm/{fn}.img.npy', img.detach().cpu().numpy())
        # np.save(f'/workspace/CVP-cotta-acdc/notebooks/S00116/losses-200mm/{fn}.loss.npy', loss_np)
        # np.save(f'/workspace/CVP-cotta-acdc/notebooks/S00116/losses-200mm/{fn}.mask.npy', mask.detach().cpu().numpy())

        losses = self._parse_losses(losses) #!DEBUG

        return losses

    def forward_train_orig(self, img, img_metas, gt_semantic_seg, seg_weight=None, return_feat=False, **kwargs):

        # feature extraction w/o mask
        x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x

        # decode loss w/o mask
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        # masking
        mask = self.mask_generator().to("cuda") #!DEBUG

        # feature extraction w/ mask
        img_latent = self.extract_feat(img, mask)

        # reconstruction
        img_rec = self.recon_neck(img_latent)

        # recon loss
        recon_loss = self.recon_head.loss(img_rec, img, mask)
        losses.update({"recon_loss": recon_loss}) #!DEBUG

        return losses


    # @staticmethod
    # def _parse_losses(losses, _lambda=1):
    #     """Parse the raw outputs (losses) of the network.

    #     Args:
    #         losses (dict): Raw output of the network, which usually contain
    #             losses and other necessary information.

    #     Returns:
    #         tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
    #             which may be a weighted sum of all losses, log_vars contains
    #             all the variables to be sent to the logger.
    #     """
    #     log_vars = OrderedDict()
    #     for loss_name, loss_value in losses.items():
    #         if isinstance(loss_value, torch.Tensor):
    #             log_vars[loss_name] = loss_value.mean()
    #         elif isinstance(loss_value, list):
    #             log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
    #         else:
    #             raise TypeError(
    #                 f'{loss_name} is not a tensor or list of tensors')

    #     a=1
    #     # loss = log_vars['decode.loss_seg'] + log_vars['recon_loss'] * _lambda #!DEBUG
    #     loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    #     log_vars['loss'] = loss
    #     for loss_name, loss_value in log_vars.items():
    #         # reduce loss when distributed training
    #         if dist.is_available() and dist.is_initialized():
    #             loss_value = loss_value.data.clone()
    #             dist.all_reduce(loss_value.div_(dist.get_world_size()))
    #         log_vars[loss_name] = loss_value.item()

    #     return loss, log_vars


    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        module = kwargs.get('module', None)
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     module=module)
        a=1 #!DEBUG
        # if len(loss_decode) == 2:
        if not isinstance(loss_decode, dict):
            loss_decode, seg_logit = loss_decode
            losses.update(add_prefix({'seg_logit': seg_logit}, 'decode'))

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses


    def forward_module(self, img, img_metas, gt_semantic_seg, return_loss=True, return_feat=False, module=4, **kwargs):
        x = self.extract_feat(img, module=module)

        losses = dict()
        if return_feat:
            losses['features'] = x

        # decode loss w/o mask
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg, module=module)
        losses.update(loss_decode)
        return losses


    def mood_demo(self, img_latent, img_metas):

        a=1
        feature_ood = self.avg_pool(img_latent[-1]).squeeze().unsqueeze(0)
        # feature_ood = feature_ood.detach().cpu().numpy()
        logit_ood = feature_ood @ self.w.T + self.b
        # softmax_ood = scipy.special.softmax(logit_ood.detach().cpu().numpy(), axis=-1)
        softmax_ood = F.softmax(logit_ood, dim=1)

        # u = -np.matmul(np.linalg.pinv(self.w), self.b)

        a=1

        img_path = img_metas[0]["filename"]
        # methods = 'MSP MaxLogit Energy'
        methods = 'MSP MaxLogit Energy ViM Residual'
        # methods = 'ViM Residual'
        fpr = 95
        _result = {
            "img_path": img_path,
            "methods": methods,
            "threshold": [],
            "results": []
        }

        # ---------------------------------------
        method = 'MSP'
        if method in methods:
            score_id = self.softmax_id_val.max(axis=-1)[0]
            score_ood = softmax_ood.max(axis=-1)[0]
            result = self.evaluate(method, score_id, score_ood, fpr)
            _result["threshold"].append(result[0])
            _result["results"].append(result[1])

        # ---------------------------------------
        method = 'MaxLogit'
        if method in methods:
            score_id = self.logit_id_val.max(axis=-1)[0]
            score_ood = logit_ood.max(axis=-1)[0]
            result = self.evaluate(method, score_id, score_ood, fpr)
            _result["threshold"].append(result[0])
            _result["results"].append(result[1])

        # ---------------------------------------
        method = 'Energy'
        if method in methods:
            score_id = torch.logsumexp(self.logit_id_val, axis=-1)
            score_ood = torch.logsumexp(logit_ood, axis=-1)
            # score_id = scipy.special.logsumexp(self.logit_id_val, axis=-1)
            # score_ood = scipy.special.logsumexp(logit_ood, axis=-1)
            result = self.evaluate(method, score_id, score_ood, fpr)
            _result["threshold"].append(result[0])
            _result["results"].append(result[1])

        # ---------------------------------------
        method = 'ViM'
        if method in methods:
            if self.feature_id_val.shape[-1] >= 2048:
                DIM = 1000 #num_cls
            elif self.feature_id_val.shape[-1] >= 768:
                DIM = 512
            else:
                DIM = self.feature_id_val.shape[-1] // 2

            # ec = EmpiricalCovariance(assume_centered=True)
            # ec.fit(self.feature_id_train - self.u)
            covariance_ = self.psuedo_empirical_covariance(self.feature_id_train)
            # eig_vals, eigen_vectors = np.linalg.eig(covariance_)
            eig_vals, eigen_vectors = torch.linalg.eig(covariance_)

            # NS = np.ascontiguousarray(
            #     (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
            NS = torch.strided(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
            vlogit_id_train = np.linalg.norm(np.matmul(self.feature_id_train - self.u, NS), axis=-1)
            alpha = self.logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()

            vlogit_id_val = np.linalg.norm(np.matmul(self.feature_id_val - self.u, NS), axis=-1) * alpha
            energy_id_val = scipy.special.logsumexp(self.logit_id_val, axis=-1)
            score_id = -vlogit_id_val + energy_id_val

            energy_ood = scipy.special.logsumexp(logit_ood, axis=-1)
            vlogit_ood = np.linalg.norm(np.matmul(feature_ood - self.u, NS), axis=-1) * alpha
            score_ood = -vlogit_ood + energy_ood
            result = self.evaluate(method, score_id, score_ood, fpr)
            _result["threshold"].append(result[0])
            _result["results"].append(result[1])

        # ---------------------------------------
        method = 'Residual'
        if method in methods:
            if self.feature_id_val.shape[-1] >= 2048:
                DIM = 1000
            elif self.feature_id_val.shape[-1] >= 768:
                DIM = 512
            else:
                DIM = self.feature_id_val.shape[-1] // 2
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(self.feature_id_train - self.u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

            score_id = -np.linalg.norm(np.matmul(self.feature_id_val - self.u, NS), axis=-1)

            score_ood = -np.linalg.norm(np.matmul(feature_ood - self.u, NS), axis=-1)
            result = self.evaluate(method, score_id, score_ood, fpr)
            _result["threshold"].append(result[0])
            _result["results"].append(result[1])
        
    #     # ---------------------------------------
    #     method = 'Novelty'
    #     if method in methods:
    #         a=1

    #     # # ---------------------------------------
    #     # method = 'Mahalanobis'
    #     # if method in methods:
    #     #     train_means = []
    #     #     train_feat_centered = []
    #     #     for i in tqdm(range(train_labels.max() + 1), desc='Computing classwise mean feature'):
    #     #         fs = self.feature_id_train[train_labels == i]
    #     #         _m = fs.mean(axis=0)
    #     #         train_means.append(_m)
    #     #         train_feat_centered.extend(fs - _m)

    #     #     ec = EmpiricalCovariance(assume_centered=True)
    #     #     ec.fit(np.array(train_feat_centered).astype(np.float64))

    #     #     mean = torch.from_numpy(np.array(train_means)).cuda().float()
    #     #     prec = torch.from_numpy(ec.precision_).cuda().float()

    #     #     score_id = -np.array(
    #     #         [(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
    #     #         for f in tqdm(torch.from_numpy(self.feature_id_val).cuda().float(),  desc='Computing Mahalanobis ID score')])

    #     #     score_ood = -np.array([
    #     #         (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
    #     #         for f in tqdm(torch.from_numpy(feature_ood).cuda().float(), desc='Computing Mahalanobis OOD score')
    #     #     ])
    #     #     result = self.evaluate(method, score_id, score_ood, fpr)


    #     if self.mood_update:
    #         self.update_features(feature_ood, logit_ood, softmax_ood)

    #     return _result

    def evaluate(self, method, score_id, score_ood, target_fpr):
        # threshold = np.percentile(score_id, 100 - target_fpr)
        threshold = torch.quantile(score_id, (100-target_fpr)*0.01)
        if score_ood >= threshold:
            print('\033[94m', method, '\033[0m', 'evaluation:', '\033[92m', 'in-distribution', '\033[0m')
        else:
            print('\033[94m', method, '\033[0m', 'evaluation:', '\033[91m', 'out-of-distribution', '\033[0m')
        return threshold, score_ood.item()


    # def psuedo_empirical_covariance(self, X):
    #     mean = torch.mean(X, dim=0)

    #     # Center the data
    #     X_centered = X - mean

    #     # Compute the covariance matrix
    #     cov_matrix = torch.matmul(X_centered.t(), X_centered) / (X_centered.size(0) - 1)

    #     return cov_matrix


class MahalanobisLayer(nn.Module):

    def __init__(self, dim, decay = 0.1):
        super(MahalanobisLayer, self).__init__()
        self.register_buffer('S', torch.eye(dim))
        self.register_buffer('S_inv', torch.eye(dim))
        self.decay = decay

    def forward(self, x, x_fit):
        """
        Calculates the squared Mahalanobis distance between x and x_fit
        """

        delta = x - x_fit
        m = torch.mm(torch.mm(delta, self.S_inv), delta.t())
        return torch.diag(m)

    def cov(self, x):
        x -= torch.mean(x, dim=0)
        return 1 / (x.size(0) - 1) * x.t().mm(x)

    def update(self, X, X_fit):
        delta = X - X_fit
        self.S = (1 - self.decay) * self.S + self.decay * self.cov(delta)
        self.S_inv = torch.pinverse(self.S)

