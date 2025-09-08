import argparse
import os

import sys
sys.path.append("...") #!TBD

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
from copy import deepcopy

from mmseg.apis.test_feat import single_gpu_our
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.ood import FreezeController
from mmseg.utils import get_root_logger
from IPython import embed

from tools.get_param_count import count_parameters

import datetime
import wandb

import numpy as np
import random
random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

DEBUG = True


def create_ema_model(model):
    ema_model = deepcopy(model)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        try:
            mcp[i].data[:] = mp[i].data[:].clone()
        except:
            mcp[i].data = mp[i].data.clone()

    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration=None):

    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--wandb', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    # set experiments
    train_serial = str(datetime.datetime.now()) if not DEBUG else "debug"
    args.show_dir = f"./cotta/exp-new/{train_serial}"
    args.out = f"./cotta/exp-new/{train_serial}/res.pkl"
    os.makedirs(args.show_dir, exist_ok=True)
    logger = get_root_logger(log_file=f"{args.show_dir}/train.log", log_level="INFO")

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:

        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg["wandb"] = args.wandb == 1 and not DEBUG
    if not DEBUG and (args.wandb == 1):
        cfg["log_config"] = dict(
            interval=1,
            hooks=[
                dict(type="TextLoggerHook", by_epoch=True),
            ],
        )

        wandb.init(
            project="Hybrid-TTA",
            name=train_serial,

            # track hyperparameters and run metadata
            config=cfg.copy()
        )

    if args.aug_test == False:
        args.aug_test = cfg.get("aug_test", False)

    if args.aug_test: #True: #args.aug_test:

        # test
        if cfg.data.test.type in ['CityscapesDataset', 'ACDCDataset']:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True

        # test1
        if cfg.data.test1.type in ['CityscapesDataset', 'ACDCDataset']:
            # hard code index
            cfg.data.test1.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test1.pipeline[1].flip = True
        elif cfg.data.test1.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test1.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test1.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test1.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test1.pipeline[1].flip = True

        # test2
        if cfg.data.test2.type in ['CityscapesDataset', 'ACDCDataset']:
            # hard code index
            cfg.data.test2.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test2.pipeline[1].flip = True
        elif cfg.data.test2.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test2.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test2.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test2.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test2.pipeline[1].flip = True

        # test3
        if cfg.data.test3.type in ['CityscapesDataset', 'ACDCDataset']:
            # hard code index
            cfg.data.test3.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test3.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test3.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test3.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test3.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test3.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    logger.info(cfg)
    datasets = [build_dataset(cfg.data.test), build_dataset(cfg.data.test1), build_dataset(cfg.data.test2),build_dataset(cfg.data.test3)]

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', logger=logger)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = True #False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    model = MMDataParallel(model, device_ids=[0])
    anchor = deepcopy(model.state_dict())
    anchor_model = deepcopy(model)
    ema_model = create_ema_model(model)

    count_parameters(model, _table=True, logger=logger)
    count_parameters(ema_model, _table=True, logger=logger)

    exp_epoch = cfg.get("exp_epoch", 10)

    freeze_ctrl = FreezeController(cfg, show_dir=args.show_dir)

    #!DEBUG
    if args.show_dir is not None:
        with open(os.path.join(args.show_dir, 'filenames.log'), 'w') as f:
            f.write('')

    for i in range(exp_epoch):
        logger.info(f"revisiting {i}")
        data_loaders = [build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False) for dataset in datasets]

        for idx, (dataset, data_loader) in enumerate(zip(datasets, data_loaders)):
            logger.info(f"[{i} - {idx}] img_dir: {dataset.img_dir}")
            outputs = single_gpu_our(model, i, data_loader, args.show, args.show_dir,
                                     efficient_test, anchor, ema_model, anchor_model,
                                     logger=logger, cfg=cfg, freeze_ctrl=freeze_ctrl)

            rank, _ = get_dist_info()
            if rank == 0:
                if args.out:
                    print(f'\nwriting results to {args.out}')
                    mmcv.dump(outputs, args.out)
                kwargs = {} if args.eval_options is None else args.eval_options
                kwargs.update({"logger": logger})
                if args.format_only:
                    dataset.format_results(outputs, **kwargs)
                if args.eval:
                    dataset.evaluate(outputs, args.eval, **kwargs)

        if cfg.data.get('eval', None):
            cfg.data.eval.test_mode = True
            eval_dataset = build_dataset(cfg.data.eval)
            eval_dataloader = build_dataloader(
                eval_dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False
            )
            logger.info(f"Source dataset evaluation: {eval_dataset.data_root}")
            kwargs = {} if args.eval_options is None else args.eval_options
            kwargs.update({"logger": logger})
            prog_bar = mmcv.ProgressBar(len(eval_dataset))

            results = []
            for i, data in enumerate(eval_dataloader):
                ema_model.eval()
                with torch.no_grad():
                    eval_result, _, _ = ema_model(return_loss=False, **data)
                    results.extend(eval_result)
                    prog_bar.update()
            eval_dataset.evaluate(results, args.eval, **kwargs)
            del eval_dataset, eval_dataloader, results

        del data_loaders

if __name__ == '__main__':
    main()
