import gc
import logging
import os
import os.path as osp
import argparse
import socket
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from common.solver.build import build_optimizer, build_scheduler
from common.utils.logger import setup_logger

from common.utils.torch_util import set_random_seed
from models.build import build_model_2d
from data.build import build_dataloader
from models.metric import per_class_iu, fast_hist
from models.utils import feature_sampling


def parse_args():
    parser = argparse.ArgumentParser(description='train source_only')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/gta_semantic_kitti/train_source_only.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def train(cfg, output_dir=''):
    logger = logging.getLogger('CoMoDaL.train_sourc_only')
    set_random_seed(cfg.RNG_SEED)
    model_2d= build_model_2d(cfg)
    model_2d = model_2d.cuda()

    optimizer_2d = build_optimizer(cfg, model_2d)
    scheduler_2d = build_scheduler(cfg, optimizer_2d)

    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    sum_period = cfg.TRAIN.SUMMARY_PERIOD
    start_iteration = 0


    set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target') if val_period > 0 else None
    ckpt = cfg.TRAIN.CHECKPOINT_PERIOD



    def setup_train():
        model_2d.train()
    def setup_validate():
        model_2d.eval()


    setup_train()
    train_iter_src = enumerate(train_dataloader_src)
    for iteration in range(start_iteration, max_iteration):
        cur_iter = iteration + 1
        _, data_batch_src = train_iter_src.__next__()
        if 'SCN' in cfg.DATASET_SOURCE.TYPE and 'SCN' in cfg.DATASET_TARGET.TYPE:
            data_batch_src['label_dense'] = data_batch_src['label_dense'].cuda()
            data_batch_src['img'] = data_batch_src['img'].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')

        optimizer_2d.zero_grad()
        sum_item = {}
        preds_2d_src = model_2d(data_batch_src)
        loss_src_2d = F.cross_entropy(preds_2d_src['seg_logit'], data_batch_src['label_dense'])

        loss_2d = loss_src_2d
        sum_item['loss_src_2d'] = loss_src_2d
        del data_batch_src,preds_2d_src
        loss_2d.backward()


        optimizer_2d.step()

        if cur_iter == 1 or (cur_iter%sum_period == 0 and sum_period>0):
            logger.info("iters:{}".format(cur_iter) + ''.join(
                (' ' + name + ':{:.4f}').format(value.item()) for name, value in sum_item.items()))
        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            setup_validate()
            sum_item_val={}
            class_miou_2d = validate(model_2d,val_dataloader,cfg)
            sum_item_val['miou_2d'] = round(np.nanmean(class_miou_2d) * 100, 2)
            sum_item_val['class_iou'] = class_miou_2d
            logger.info(''.join(
                (name + ':{}' + ' ').format(value) for name, value in sum_item_val.items()))

            setup_train()

        # ---------------------------------------------------------------------------- #
        scheduler_2d.step()

        if (ckpt > 0 and cur_iter % ckpt == 0) or cur_iter == max_iteration:
            filepath = os.path.join(output_dir, 'model_2d_{}.pth'.format(cur_iter))
            torch.save(model_2d.state_dict(), filepath)
            print('save model_2d_{}'.format(cur_iter))
        gc.collect()
        torch.cuda.empty_cache()
def validate(model_2d,dataloader,cfg):
    hist_2d = np.zeros((cfg.MODEL_2D.NUM_CLASSES, cfg.MODEL_2D.NUM_CLASSES))
    with torch.no_grad():
        for iteration, data_batch in enumerate(tqdm(dataloader)):
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            preds_2d = model_2d(data_batch)
            seg_logit_2d = feature_sampling(preds_2d['seg_logit'], data_batch['img_indices'])
            seg_label = data_batch['seg_label']
            pred_label_2d = seg_logit_2d.argmax(1)

            hist_2d += fast_hist(np.array(seg_label.cpu()).flatten(), np.array(pred_label_2d.cpu()).flatten(),
                                 cfg.MODEL_2D.NUM_CLASSES)
        inters_over_union_classes_2d = per_class_iu(hist_2d)
    return inters_over_union_classes_2d





def main():
    args = parse_args()

    from common.config import purge_cfg
    from config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)
    logger = setup_logger('CoMoDaL', output_dir, comment='train_source_only.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))
    train(cfg, output_dir)


if __name__ == '__main__':
    main()