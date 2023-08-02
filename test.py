import logging
import os
import os.path as osp
import argparse
import socket
import time
import warnings
import numpy as np
import torch
from tqdm import tqdm
from common.utils.logger import setup_logger
from common.utils.torch_util import set_random_seed
from models.build import build_model_2d, build_model_3d
from data.build import build_dataloader
from models.metric import per_class_iu, fast_hist
from models.utils import feature_sampling


def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/gta_semantic_kitti/test.yaml',
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


def test(cfg):
    logger = logging.getLogger('CoMoDaL.test')
    set_random_seed(cfg.RNG_SEED)
    model_2d = build_model_2d(cfg)
    model_3d = build_model_3d(cfg)
    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()
    model_2d.load_state_dict(torch.load(cfg.CHECKPOINT2D_DIR),strict=False)
    model_3d.load_state_dict(torch.load(cfg.CHECKPOINT3D_DIR),strict=False)
    set_random_seed(cfg.RNG_SEED)
    test_dataloader = build_dataloader(cfg, mode='test', domain='target')

    def setup_validate():
        model_2d.eval()
        model_3d.eval()

    setup_validate()
    sum_item_val={}
    class_miou_2d,class_miou_3d,class_miou_avg = validate(model_2d,model_3d,test_dataloader,cfg)
    sum_item_val['miou_2d'] = round(np.nanmean(class_miou_2d) * 100, 2)
    sum_item_val['miou_3d'] = round(np.nanmean(class_miou_3d) * 100, 2)
    sum_item_val['miou_avg'] = round(np.nanmean(class_miou_avg) * 100, 2)
    sum_item_val['class_iou_2d'] = class_miou_2d
    sum_item_val['class_iou_3d'] = class_miou_3d
    sum_item_val['class_iou_avg'] = class_miou_avg
    logger.info(''.join(
        (name + ':{}' + ' ').format(value) for name, value in sum_item_val.items()))

def validate(model_2d,model_3d,dataloader,cfg):
    hist_2d = np.zeros((cfg.MODEL_2D.NUM_CLASSES, cfg.MODEL_2D.NUM_CLASSES))
    hist_3d = np.zeros((cfg.MODEL_3D.NUM_CLASSES, cfg.MODEL_3D.NUM_CLASSES))
    hist_avg = np.zeros((cfg.MODEL_3D.NUM_CLASSES, cfg.MODEL_3D.NUM_CLASSES))
    with torch.no_grad():
        for iteration, data_batch in enumerate(tqdm(dataloader)):
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError
            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch)
            seg_logit_2d = feature_sampling(preds_2d['seg_logit'], data_batch['img_indices'])
            seg_logit_3d = preds_3d['seg_logit']
            seg_label = data_batch['seg_label']
            pred_label_2d = seg_logit_2d.argmax(1)
            pred_label_3d = seg_logit_3d.argmax(1)
            pred_label_avg = (torch.softmax(seg_logit_2d, dim=1) + torch.softmax(seg_logit_3d, dim=1)).argmax(1)

            hist_2d += fast_hist(np.array(seg_label.cpu()).flatten(), np.array(pred_label_2d.cpu()).flatten(),
                                 cfg.MODEL_2D.NUM_CLASSES)
            hist_3d += fast_hist(np.array(seg_label.cpu()).flatten(), np.array(pred_label_3d.cpu()).flatten(),
                                 cfg.MODEL_3D.NUM_CLASSES)
            hist_avg += fast_hist(np.array(seg_label.cpu()).flatten(), np.array(pred_label_avg.cpu()).flatten(),
                                  cfg.MODEL_3D.NUM_CLASSES)
        inters_over_union_classes_2d = per_class_iu(hist_2d)
        inters_over_union_classes_3d = per_class_iu(hist_3d)
        inters_over_union_classes_avg = per_class_iu(hist_avg)
    return inters_over_union_classes_2d,inters_over_union_classes_3d,inters_over_union_classes_avg



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
    logger = setup_logger('CoMoDaL', output_dir, comment='test.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))
    test(cfg)


if __name__ == '__main__':
    main()