import torch
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.utils import freeze

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

from typing import List
from collections import OrderedDict


@SEGMENTORS.register_module()
class OthersEncoderDecoder(EncoderDecoder):
    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        **cfg
        ):
        super(OthersEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            # init_cfg=init_cfg,
        )
        num_module = 4
        self.decodesc_flag = "decoder_custom" in backbone

        assert self.backbone
        assert self.decode_head

    def get_main_model(self):
        return self.main_model

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if kwargs.get('module', None):
            return self.forward_module(img, img_metas, return_loss=return_loss, **kwargs)
        else:
            return super().forward(img, img_metas, return_loss=return_loss, **kwargs)

    def forward_module(self, img, img_metas, gt_semantic_seg, return_loss=True, return_feat=False, module=4, **kwargs):
        x = self.extract_feat(img, module=module)

        losses = dict()
        if return_feat:
            losses['features'] = x

        # decode loss w/o mask
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg, module=module)
        losses.update(loss_decode)
        return losses

    def extract_feat(self, img, **kwargs):
        if kwargs.get('module', None) is not None:
            x = self.backbone.forward_module(img, module=kwargs['module'])
            if self.with_neck:
                x = self.neck(x)
            return x
        else:
            return super().extract_feat(img)

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        if kwargs.get('module', None) is not None:
            loss_decode = self.decode_head.forward_train(x, img_metas,
                                                         gt_semantic_seg,
                                                         self.train_cfg,
                                                         module=kwargs['module'])
        else:
            loss_decode = self.decode_head.forward_train(x, img_metas,
                                                         gt_semantic_seg,
                                                         self.train_cfg)

        if not isinstance(loss_decode, dict):
            loss_decode, seg_logit = loss_decode
            losses.update(add_prefix({'seg_logit': seg_logit}, 'decode'))

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def extract_feat_decodesc(self, img):
        x, c = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        return x, c
    
    def _decode_head_forward_train_decodesc(self, x, c, img_metas,
                                            gt_semantic_seg, seg_weight=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, c, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     seg_weight)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    
    def _decode_head_forward_test_decodesc(self, x, c, img_metas):
        seg_logits = self.decode_head.forward_test(x, c, img_metas, self.test_cfg)
        return seg_logits

    def forward_train(self, img, img_metas, gt_semantic_seg, seg_weight=None, return_feat=False, **kwargs):
        if self.decodesc_flag:
            x, c = self.extract_feat_decodesc(img)
        else:
            x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x

        if self.decodesc_flag:
            loss_decode = self._decode_head_forward_train_decodesc(x, c, img_metas,
                                                                   gt_semantic_seg, seg_weight)
        else:
            # loss_decode = self._decode_head_forward_train(x, img_metas,
            #                                               gt_semantic_seg, seg_weight)
            loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)
        
        losses = self._parse_losses(losses)

        return losses
    
    def encode_decode(self, img, img_metas):
        if self.decodesc_flag:
            x, c = self.extract_feat_decodesc(img)
            out = self._decode_head_forward_test_decodesc(x, c, img_metas)
        else:
            x = self.extract_feat(img)
            out = self._decode_head_forward_test(x, img_metas)

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def entropy_prediction(self, img):
        x = self.extract_feat(img)

        entr, conf = self.decode_head.calculate_entropy(x)

        return {f"confidence": conf, "entropy": entr}

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

    def freeze_components(self, modules: List[str], except_keywords: List[str]=[]):
        if len(modules) == 0: return
        for name, param in self.named_parameters():
            for _module in modules:
                if _module in name:
                    flags = [(x in name) for x in except_keywords]
                    if sum(flags) == 0:
                        param.requires_grad_(False)
                        param.grad = None

    def unfreeze_components(self, modules: List[str], **kwargs):
        if len(modules) == 0: return
        for name, param in self.named_parameters():
            for _module in modules:
                if _module in name:
                    param.requires_grad_(True)


    @staticmethod
    def _parse_losses(losses):
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

