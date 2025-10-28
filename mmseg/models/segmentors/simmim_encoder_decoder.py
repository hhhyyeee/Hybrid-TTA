import torch
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.utils import freeze

from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

from typing import List
import numpy as np
np.random.seed(42)
from collections import OrderedDict


class MaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        assert isinstance(input_size, tuple)

        H, W = input_size
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert H % self.mask_patch_size == 0
        assert W % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = np.zeros(2, dtype=int)
        self.rand_size[0] = self.input_size[0] // self.mask_patch_size
        self.rand_size[1] = self.input_size[1] // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size[0] * self.rand_size[1]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, bulk=False):
        if not bulk:
            mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        else:
            pass
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size[0], self.rand_size[1]))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        mask = torch.from_numpy(mask).unsqueeze(0) #!DEBUG

        return mask


@SEGMENTORS.register_module()
class SimMIMEncoderDecoder(EncoderDecoder):
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
        **cfg
        ):
        super(SimMIMEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        self.recon_neck = builder.build_neck(cfg["recon_neck"])
        self.recon_head = builder.build_head(cfg["recon_head"])
        self.recon_lambda = cfg.get('recon_lambda', 1.)

        self.mask_generator = MaskGenerator(
            # input_size=1024,
            # input_size=mask_cfg["test_input_size"], #720,
            input_size=mask_cfg["img_size"],
            mask_patch_size=32,
            model_patch_size=4,
            # mask_ratio=0.6)
            mask_ratio=mask_cfg["mask_ratio"])

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        # simmim

        # segmentation
        if return_loss:
            mim_mode = kwargs.get("mim_mode", "orig")
            if mim_mode == "simmim":
                return self.forward_train_simmim(img, img_metas, **kwargs)
            elif mim_mode == "orig":
                return self.forward_train_orig(img, img_metas, **kwargs)
            else:
                assert mim_mode in ["simmim", "orig"]
        else:
            return self.forward_test(img, img_metas, **kwargs)


    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)

        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)

        #!DEBUG
        prob, pred = seg_logit.max(dim=1)
        prob = [prob.cpu().numpy()]
        pred = [pred.cpu().numpy()]

        return seg_pred, prob, pred

    def extract_feat(self, img, mask=None):
        """Extract features from images."""
        x = self.backbone(img, mask=mask)
        if self.with_neck:
            x = self.neck(x)
        return x


    def forward_train_simmim(self, img, img_metas, gt_semantic_seg, hog=None, seg_weight=None, return_feat=False, **kwargs):

        # masking
        mask = self.mask_generator().to("cuda") #!DEBUG

        # feature extraction w/ mask
        img_latent = self.extract_feat(img, mask)
        losses = dict()
        if return_feat:
            losses['features'] = img_latent

        # reconstruction
        img_rec = self.recon_neck(img_latent)

        # recon loss
        recon_loss = self.recon_head.loss(img_rec, img, mask)
        losses.update({"recon_loss": recon_loss}) #!DEBUG

        # decode loss w/ mask
        loss_decode = self._decode_head_forward_train(img_latent, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        losses = self._parse_losses(losses)

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

    def freeze_components(self, modules: List[str], except_keywords: List[str]=[]):
        if len(modules) == 0: return
        for name, param in self.named_parameters():
            for _module in modules:
                if _module in name:

                    flags = [(x in name) for x in except_keywords]
                    if sum(flags) == 0:
                        param.requires_grad_(False)
                        param.grad = None
                    else:
                        pass

    def unfreeze_components(self, modules: List[str], **kwargs):
        if len(modules) == 0: return
        for name, param in self.named_parameters():
            for _module in modules:
                if _module in name:
                    param.requires_grad_(True)


    # @staticmethod
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        if self.recon_lambda != 1.:
            loss = log_vars['decode.loss_seg'] + log_vars['recon_loss'] * self.recon_lambda #!DEBUG
        else:
            loss = sum(_value for _key, _value in log_vars.items()
                    if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

