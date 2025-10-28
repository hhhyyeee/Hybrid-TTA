import os
import os.path as osp
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmcv.runner import build_optimizer, build_runner

from tools.get_param_count import count_parameters, count_all_parameters

from IPython import embed
from mmseg.ops import resize

import numpy as np
import kornia
import torch
import random
import torch.nn as nn
import wandb

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
import pdb


def update_ema_variables(ema_model, model, alpha_teacher, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    for (ema_name, ema_param), (name, param) in zip(ema_model.named_parameters(), model.named_parameters()):
        if not torch.equal(ema_param, param):
            pass

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

            try:
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
            except:
                ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
    return ema_model

def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir='./tmp').name
    np.save(temp_file_name, array)
    return temp_file_name

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def single_gpu_our(model,
                   epoch_index,
                   data_loader,
                   show=False,
                   out_dir=None,
                   efficient_test=False,
                   anchor=None,
                   ema_model=None,
                   anchor_model=None,
                   show_interval=10,
                   logger=None,
                   cfg=None,
                   **kwargs):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:  
        list: The prediction results.
    """

    model.eval()
    anchor_model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    # if out_dir is None:
    #     out_dir = "./cotta/"+str(datetime.datetime.now())

    log_json_path = f'{out_dir}/log.json'
    Path(log_json_path).touch()

    # FreezeController
    freeze_ctrl = kwargs.get('freeze_ctrl', None)
    assert freeze_ctrl is not None
    freeze_ctrl.reset(model) # unfreeze model before starting every epoch
    if freeze_ctrl.mode == 'manual':
        freeze_ctrl(model=model, epoch_index=epoch_index)
    if freeze_ctrl.mode == 'random_test':
        freeze_ctrl(model=model, epoch_index=epoch_index)

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
            # print(name) #!DEBUG (log 파일이 너무 시끄러움)
        else:
            param.requires_grad=False

    #!DEBUG
    print(count_all_parameters(model))
    print(count_parameters(model))

    # optimizer = torch.optim.Adam(param_list, lr=0.00006/8, betas=(0.9, 0.999))#Batchsize=1 now, was 8 during cityscapes training
    optimizer = torch.optim.Adam(param_list, lr=cfg["optimizer"]["lr"]/8, betas=(0.9, 0.999))
    # optimizer = torch.optim.AdamW(param_list, lr=cfg["optimizer"]["lr"]/8, betas=(0.9, 0.999))
    if epoch_index >= cfg.get('freeze_epoch', 10):
    # if epoch_index >= freeze_ctrl.freeze_epoch:
        optimizer = torch.optim.Adam(param_list, lr=cfg["optimizer"]["lr"]/8/10, betas=(0.9, 0.999))
        # optimizer = torch.optim.AdamW(param_list, lr=cfg["optimizer"]["lr"]/8/10, betas=(0.9, 0.999))
        logger.info(f"PEFT optimizer learning rate: {optimizer.param_groups[0]['lr']}")

    # scheduler
    _scheduler = cfg["lr_config"].get("scheduler", None)
    if _scheduler == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16000, eta_min=0, last_epoch=-1)
    elif _scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["optimizer"]["lr"]/8,
                                                        total_steps=16000,
                                                        anneal_strategy=cfg["lr_config"]["anneal_strategy"])
    elif _scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_config"]["step_size"])
    else:
        scheduler = None

    # wandb
    _wandb = cfg.get('wandb', False)

    softmax_entropy = 0.0 #!DEBUG
    for i, data in enumerate(data_loader):
        _start = time.time()
        if i % show_interval == 0:
            logger.info(f"[{i} / {len(data_loader)}]")
        wandb_log = {}

        if freeze_ctrl.mode == 'manual':
            freeze_ctrl(model=model, epoch_index=epoch_index, iter_index=i)
            wandb_log.update({'freeze_ctrl.frozen_flag': int(freeze_ctrl.frozen_flag)})

        model.eval()
        ema_model.eval()
        anchor_model.eval()

        elapsed_time = time.time() - _start

        with torch.no_grad():
            IMG_ID = 4 if len(data["img"]) > 1 else 0

            if True:
                anchor_feat = anchor_model.module.extract_feat(data['img'][IMG_ID].cuda())
                student_feat = model.module.extract_feat(data['img'][IMG_ID].cuda())
                teacher_feat = ema_model.module.extract_feat(data['img'][IMG_ID].cuda())

                if False and (i % 5 == 0):
                    os.makedirs(f'{out_dir}/tsne/', exist_ok=True)
                    domain = data['img_metas'][0].data[0][0]['filename'].split('/')[6]
                    with open(f'{out_dir}/tsne/feat_epoch{epoch_index}_{domain}_iter{i}.pickle', 'wb') as f:
                        pickle.dump([x.detach().cpu() for x in teacher_feat], f)

                fdist_dict = {}
                for fi, (af, sf, tf) in enumerate(zip(anchor_feat, student_feat, teacher_feat)):
                    af = af[0].detach()
                    sf = sf[0].detach()
                    tf = tf[0].detach()

                    asf = torch.mean(torch.norm(af - sf, dim=1, p=2))
                    atf = torch.mean(torch.norm(af - tf, dim=1, p=2))
                    stf = torch.mean(torch.norm(sf - tf, dim=1, p=2))

                    fdist_dict[f'fdist.anchor_student.{fi}'] = asf.item()
                    fdist_dict[f'fdist.anchor_teacher.{fi}'] = atf.item()
                    fdist_dict[f'fdist.student_teacher.{fi}'] = stf.item()
                wandb_log.update(fdist_dict)

            _start = time.time()
            # model inference
            student_result, _, _ = model(return_loss=False, **data)

            # ema model inference
            result, _, preds = ema_model(return_loss=False, **data)

            if isinstance(result, list):
                if len(data["img"]) > 1:
                    # anchor model inference
                    _, probs_, _ = anchor_model(return_loss=False, **data)

                    mask = (probs_[IMG_ID][0] > 0.69).astype(np.int64) # 0.74 was the 5% quantile for cityscapes, therefore we use 0.69 here
                    # anchor confidance 높은부분은 preds 사용, 낮은 부분은 results 사용
                    # result: 모든 augmentations 평균 낸 output. len(result)는 0이고 result[0].shape은 (1080, 1920)
                    # preds: 모든 augmentations들의 output list. len(preds)는 6 또는 14이고 preds[4].shape은 (1, 1080, 1920)
                    result = [(mask*preds[IMG_ID][0] + (1.-mask)*result[0]).astype(np.int64)]
                    weight = 1.
                else:
                    weight = 1.
            else:
                continue
            elapsed_time += time.time() - _start

        # visualize
        if (show or out_dir) and (i % show_interval == 0):

            with open(osp.join(out_dir, 'filenames.log'), 'a') as f:
                f.write(f"[{epoch_index:3} - {i:4}] {data['img_metas'][0].data[0][0]['filename']}\n")

            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    #!DEBUG
                    import re
                    filename = data['img_metas'][0].data[0][0]['filename']
                    domain = re.search(f'\d*mm', filename)

                    img_filename = img_meta["ori_filename"].replace(".png", ".pseudo.png")

                    #!DEBUG
                    if domain is not None:
                        domain = domain.group()
                        img_filename = img_filename.replace(".png", f".{domain}.png")

                    out_file = osp.join(out_dir, img_filename)
                else:
                    out_file = None

                # ema model output
                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

                # student output
                if out_file:
                    out_file = out_file.replace(".pseudo", ".student")
                model.module.show_result(
                    img_show,
                    student_result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        del student_result

        # back propagate
        if isinstance(result, list):
            _start = time.time()
            USE_GT = cfg.get('use_gt', False)
            if USE_GT:
                loss = model.forward(
                    return_loss=True,
                    img=data['img'][IMG_ID],
                    img_metas=data['img_metas'][IMG_ID].data[0],
                    gt_semantic_seg=data['gt_semantic_seg'][0].type(torch.long).cuda(),
                    **cfg)
            else:
                loss = model.forward(
                    return_loss=True,
                    img=data['img'][IMG_ID],
                    img_metas=data['img_metas'][IMG_ID].data[0],
                    gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0),
                    **cfg)
            elapsed_time += time.time() - _start

            #!DEBUG (visualize reconstructed images)
            recons = model.module.misc.get('img_rec', None)
            if (show or out_dir) and (i % show_interval == 0) and (recons is not None):

                img_metas = data['img_metas'][0].data[0]
                recons = tensor2imgs(recons, **img_metas[0]['img_norm_cfg'])

                for recon, img_meta in zip(recons, img_metas):

                    h, w, _ = img_meta['img_shape']
                    recon_show = recon[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    recon_show = mmcv.imresize(recon_show, (ori_w, ori_h))

                    if out_dir:
                        img_filename = img_meta['ori_filename'].replace('.png', '.recon.png')
                        out_file = osp.join(out_dir, img_filename)
                    else:
                        out_file = None

                    mmcv.imwrite(recon_show, out_file)

            _start = time.time()
            if freeze_ctrl.mode == 'loss_seg':
                freeze_ctrl(model, epoch_index=epoch_index, iter_index=i, loss=loss)
                wandb_log.update({'freeze_ctrl.threshold': freeze_ctrl.threshold, 'freeze_ctrl.frozen_flag': int(freeze_ctrl.frozen_flag)})
            if freeze_ctrl.mode == 'loss_seg_with_buffer':
                freeze_ctrl(model, epoch_index=epoch_index, iter_index=i, loss=loss)
                wandb_log.update({'freeze_ctrl.threshold': freeze_ctrl.threshold, 'freeze_ctrl.frozen_flag': int(freeze_ctrl.frozen_flag)})
            if freeze_ctrl.mode == 'loss_seg_with_wma':
                freeze_ctrl(model, epoch_index=epoch_index, iter_index=i, loss=loss)
                wandb_log.update({
                    'freeze_ctrl.threshold': freeze_ctrl.threshold,
                    'freeze_ctrl.frozen_flag': int(freeze_ctrl.frozen_flag),
                    'freeze_ctrl.wma_loss': freeze_ctrl.wma_loss})

            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
            elapsed_time += time.time() - _start
        else:
            #list가 아니면 역전파가 없음?
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        _start = time.time()
        # loss, log_vars = model.module._parse_losses(loss)
        if len(loss) == 2:
            loss, log_vars = loss
        torch.mean(weight*loss).backward()
        # torch.mean(weight*loss["decode.loss_seg"]).backward()
        wandb_log.update(log_vars)
        if log_vars.get('softmax_entropy', None) and freeze_ctrl.mode == 'entropy':
            softmax_entropy = log_vars['softmax_entropy']
            # logger.info(f"softmax_entropy: {softmax_entropy:.5f}")
            freeze_ctrl(model, epoch_index=epoch_index, iter_index=i,
                        loss=loss, log_vars=log_vars, softmax_entropy=softmax_entropy)

        optimizer.step()
        optimizer.zero_grad()
        elapsed_time = time.time() - _start

        if scheduler is not None:
            scheduler.step()
        if i % show_interval == 0:
            if scheduler is not None:
                logger.info(f"lr (scheduler): {scheduler.get_last_lr()[0]}")
                if _wandb: wandb_log.update({"lr": scheduler.get_last_lr()[0]})
            else:
                logger.info(f"lr (optimizer): {optimizer.param_groups[0]['lr']}")
                if _wandb: wandb_log.update({"lr": optimizer.param_groups[0]['lr']})

        del loss
        torch.cuda.empty_cache()

        # ema model update
        _start = time.time()
        alpha_teacher = cfg.get('alpha_teacher', 0.999) #!DEBUG
        ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_teacher=alpha_teacher)
        elapsed_time += time.time() - _start
        wandb_log.update({'elapsed_time': elapsed_time})

        def cos_sims(model, ema_model, anchor_model, name=''):
            def model_params_view(model):
                params = []
                for param in model.parameters():
                    params.append(param.view(-1))
                params = torch.cat(params)
                return params

            model_params = model_params_view(model)
            ema_model_params = model_params_view(ema_model)
            anchor_model_params = model_params_view(anchor_model)

            cos = torch.nn.CosineSimilarity(dim=0)
            if name == '':
                _log = {
                    'cos.model.ema_model': cos(model_params, ema_model_params),
                    'cos.model.anchor_model': cos(model_params, anchor_model_params),
                    'cos.ema_model.anchor_model': cos(ema_model_params, anchor_model_params),
                }
            else:
                _log = {
                    f'cos.model.ema_model.{name}': cos(model_params, ema_model_params),
                    f'cos.model.anchor_model.{name}': cos(model_params, anchor_model_params),
                    f'cos.ema_model.anchor_model.{name}': cos(ema_model_params, anchor_model_params),
                }
            return _log
        wandb_log.update(cos_sims(model.module.backbone, ema_model.module.backbone, anchor_model.module.backbone))
        wandb_log.update(cos_sims(model.module.backbone.block1, ema_model.module.backbone.block1, anchor_model.module.backbone.block1, 'backbone_block1'))
        wandb_log.update(cos_sims(model.module.backbone.block2, ema_model.module.backbone.block2, anchor_model.module.backbone.block2, 'backbone_block2'))
        wandb_log.update(cos_sims(model.module.backbone.block3, ema_model.module.backbone.block3, anchor_model.module.backbone.block3, 'backbone_block3'))
        wandb_log.update(cos_sims(model.module.backbone.block4, ema_model.module.backbone.block4, anchor_model.module.backbone.block4, 'backbone_block4'))
        wandb_log.update(cos_sims(model.module.decode_head, ema_model.module.decode_head, anchor_model.module.decode_head, 'decode_head'))
        try:
            wandb_log.update(cos_sims(model.module.recon_neck, ema_model.module.recon_neck, anchor_model.module.recon_neck, 'recon_neck'))
        except:
            pass

        # wandb logging
        if _wandb:
            wandb.log(wandb_log)
        with open(log_json_path, 'a') as f:
            f.write(str(wandb_log) + '\n')

        # stochastic restoration
        cotta_flag = True if cfg["model"]["type"] == "EncoderDecoder" else False
        if cotta_flag:
            for nm, m  in model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.01).float().cuda() 
                        with torch.no_grad():
                            p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return results

