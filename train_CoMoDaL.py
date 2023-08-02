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
import torchvision
from tqdm import tqdm

from common.solver.build import build_optimizer, build_scheduler
from common.utils.logger import setup_logger

from common.utils.torch_util import set_random_seed
from data.utils.augmentation_3d import augment_and_scale_3d
from models.build import build_model_2d, build_model_3d
from data.build import build_dataloader
from models.metric import per_class_iu, fast_hist
from models.utils import feature_sampling, get_feature, get_feature_,get_masks


def parse_args():
    parser = argparse.ArgumentParser(description='train ICD+ICG')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/gta_semantic_kitti/train_CoMoDaL.yaml',
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

def generate_mixbatch(cfg,data_batch_src,data_batch_trg,seg_logit_3d,pl_2d,pl_3d):
    data_batch_mix = {}
    batch, channel, height, width = data_batch_src['orig_img'].shape
    locs_batch = []
    feats_batch = []
    data_batch_mix['img'] = torch.zeros([batch, channel, height, width]).cuda()
    data_batch_mix['seg_label_2d'] = torch.zeros([batch, height, width]).cuda()
    data_batch_mix['img_indices'] = []
    data_batch_mix['pl'] = []
    data_batch_mix['seg_logit_3d'] = []
    data_batch_mix['mask'] = []
    mix_mask = get_masks(img=data_batch_src['orig_img'], mask_type='cut')
    rand_index = torch.randperm(cfg.TRAIN.BATCH_SIZE)
    a_ = 0
    for i in range(cfg.TRAIN.BATCH_SIZE):

        b_ = a_
        a_ += len(data_batch_trg['orig_img_indices'][i])
        mix_mask_2d_src = mix_mask[i]
        mask_idx = (data_batch_trg['orig_x'][0][:, 3] == rand_index[i])

        data_batch_mix['img'][i] = data_batch_src['orig_img'][i] * mix_mask_2d_src + (1 - mix_mask_2d_src) * \
                                   data_batch_trg['orig_img'][i]

        data_batch_mix['seg_label_2d'][i] = data_batch_src['orig_label_dense'][i] * mix_mask_2d_src
        mask = (feature_sampling(mix_mask[i], [data_batch_trg['orig_img_indices'][i]]).squeeze() == 0).cpu()

        color_jitter = torchvision.transforms.ColorJitter(*(0.4, 0.4, 0.4))
        data_batch_mix['img'][i] = color_jitter(data_batch_mix['img'][i])
        indices = data_batch_trg['orig_img_indices'][i]
        if np.random.rand() < 0.5:
            data_batch_mix['img'][i] = torch.fliplr(data_batch_mix['img'][i].permute(1, 2, 0)).permute(2, 0, 1)
            data_batch_mix['seg_label_2d'][i] = torch.fliplr(data_batch_mix['seg_label_2d'][i])
            indices[:, 1] = width - 1 - indices[:, 1]
        points_src = data_batch_trg['orig_points'][i]
        points_trg = data_batch_trg['orig_points'][rand_index[i]]

        coords = augment_and_scale_3d(points=np.concatenate([points_src, points_trg], axis=0), scale=20,
                                      full_scale=4096, noisy_rot=0.1, flip_y=0.5,
                                      rot_z=6.2831, transl=True)

        coords = coords.astype(np.int64)
        # only use voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < 4096)
        coords = torch.from_numpy(coords[idxs])
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(i)
        locs_batch.append(torch.cat([coords, batch_idxs], 1))
        feats = torch.from_numpy(np.ones([len(idxs), 1], np.float32))
        feats_batch.append(feats)
        data_batch_mix['img_indices'].append(indices)
        data_batch_mix['pl'].append(torch.cat([pl_2d[b_:a_], pl_3d[mask_idx]])[idxs])

        data_batch_mix['seg_logit_3d'].append(seg_logit_3d[b_:a_][mask])
        data_batch_mix['mask'].append(mask)
    data_batch_mix['mask'] = torch.cat(data_batch_mix['mask'])
    locs_batch = torch.cat(locs_batch, dim=0)
    feats_batch = torch.cat(feats_batch, dim=0).cuda()
    data_batch_mix['x'] = [locs_batch, feats_batch]
    data_batch_mix['pl'] = torch.cat(data_batch_mix['pl'], dim=0)
    data_batch_mix['seg_logit_3d'] = torch.cat(data_batch_mix['seg_logit_3d'], dim=0)

    return data_batch_mix
def train(cfg, output_dir=''):
    logger = logging.getLogger('CoMoDaL.train_CoMoDaL')
    set_random_seed(cfg.RNG_SEED)
    model_2d = build_model_2d(cfg)
    model_3d = build_model_3d(cfg)
    model_2d_pretrain = build_model_2d(cfg)
    model_3d_ema = build_model_3d(cfg)
    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()
    model_2d_pretrain = model_2d_pretrain.cuda()
    model_2d_pretrain.load_state_dict(torch.load(cfg.PRETRAIN_DIR),
                                      strict=False)
    model_3d_ema = model_3d_ema.cuda()
    model_3d_ema.load_state_dict(model_3d.state_dict())
    optimizer_2d = build_optimizer(cfg, model_2d)
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    optimizer_3d = build_optimizer(cfg, model_3d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    sum_period = cfg.TRAIN.SUMMARY_PERIOD
    start_iteration = 0

    set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration)
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target') if val_period > 0 else None
    ckpt = cfg.TRAIN.CHECKPOINT_PERIOD

    def setup_train():
        model_2d.train()
        model_3d.train()

    def setup_validate():
        model_2d.eval()
        model_3d.eval()



    setup_train()
    model_2d_pretrain.eval()
    model_3d_ema.eval()
    train_iter_src = enumerate(train_dataloader_src)
    train_iter_trg = enumerate(train_dataloader_trg)
    for iteration in range(start_iteration, max_iteration):
        cur_iter = iteration + 1
        _, data_batch_src = train_iter_src.__next__()
        _, data_batch_trg = train_iter_trg.__next__()
        if 'SCN' in cfg.DATASET_SOURCE.TYPE and 'SCN' in cfg.DATASET_TARGET.TYPE:
            # source
            data_batch_src['label_dense'] = data_batch_src['label_dense'].cuda()
            data_batch_src['img'] = data_batch_src['img'].cuda()
            data_batch_trg['img'] = data_batch_trg['img'].cuda()
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()
        sum_item = {}


        preds_2d_src = model_2d(data_batch_src)
        loss_src_2d = F.cross_entropy(preds_2d_src['seg_logit'], data_batch_src['label_dense'])
        loss_2d = loss_src_2d
        sum_item['loss_src_2d'] = loss_src_2d
        loss_2d.backward()
        del preds_2d_src
        del data_batch_src['img']

        preds_2d_trg = model_2d(data_batch_trg)
        preds_3d_trg = model_3d(data_batch_trg)
        with torch.no_grad():
            preds_2d_trg_pretrain = model_2d_pretrain(data_batch_trg)
            prob_p, pl_p = torch.max(feature_sampling(torch.softmax(preds_2d_trg_pretrain['seg_logit'].detach(), dim=1),data_batch_trg['img_indices']), dim=1)
            prob_o, pl_o = torch.max(feature_sampling(torch.softmax(preds_2d_trg['seg_logit'].detach(), dim=1),data_batch_trg['img_indices']), dim=1)
            pl_2d = pl_p
            pl_2d[prob_p < prob_o] = pl_o[prob_p < prob_o]
        seg_logit_2d = preds_2d_trg['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d_trg['seg_logit']
        seg_logit_3d = preds_3d_trg['seg_logit']
        loss_trg_2d = F.kl_div(
            F.log_softmax(feature_sampling(seg_logit_2d, data_batch_trg['img_indices']), dim=1),
            F.softmax(preds_3d_trg['seg_logit'].detach(), dim=1),
            reduction='none').sum(1).mean()
        loss_trg_3d =  F.cross_entropy(seg_logit_3d, pl_2d)
        loss_2d = cfg.TRAIN.XMUDA.lambda_trg_2d * loss_trg_2d
        loss_3d = loss_trg_3d
        sum_item['loss_trg_2d'] = loss_trg_2d
        sum_item['loss_trg_3d'] = loss_trg_3d
        del preds_2d_trg
        del preds_2d_trg_pretrain
        del data_batch_trg['img'], data_batch_trg['x'], data_batch_trg['img_indices']
        del preds_3d_trg
        loss_2d.backward()
        loss_3d.backward()

        if cfg.TRAIN.XMUDA.lambda_mix_2d >0 and cfg.TRAIN.XMUDA.lambda_mix_3d >0:

            data_batch_src['orig_label_dense'] = data_batch_src['orig_label_dense'].cuda()
            data_batch_src['orig_img'] = data_batch_src['orig_img'].cuda()
            # target
            data_batch_trg['orig_img'] = data_batch_trg['orig_img'].cuda()
            data_batch_trg['orig_x'][1] = data_batch_trg['orig_x'][1].cuda()
            with torch.no_grad():
                data_batch_trg['img'] = data_batch_trg['orig_img']
                data_batch_trg['img_indices'] = data_batch_trg['orig_img_indices']
                data_batch_trg['x'] = data_batch_trg['orig_x']
                preds_2d_pretrain = model_2d_pretrain(data_batch_trg)
                preds_2d = model_2d(data_batch_trg)
                preds_3d = model_3d(data_batch_trg)
                preds_3d_ema = model_3d_ema(data_batch_trg)
                prob_p_, pl_p_ = torch.max(feature_sampling(torch.softmax(preds_2d_pretrain['seg_logit'].detach(), dim=1),
                                                         data_batch_trg['img_indices']), dim=1)
                prob_o_, pl_o_ = torch.max(feature_sampling(torch.softmax(preds_2d['seg_logit'].detach(), dim=1),
                                                         data_batch_trg['img_indices']), dim=1)
                pl_2d_ = pl_p_
                pl_2d_[prob_p < prob_o] = pl_o_[prob_p < prob_o]
                pl_3d_ = preds_3d_ema['seg_logit'].argmax(1)
                seg_logit_3d = preds_3d['seg_logit']

            del preds_3d,preds_2d,preds_2d_pretrain,preds_3d_ema
            data_batch_mix = generate_mixbatch(cfg,data_batch_src,data_batch_trg, seg_logit_3d,pl_2d_,pl_3d_)

            preds_2d_mix = model_2d(data_batch_mix)
            preds_3d_mix = model_3d(data_batch_mix)
            # for target pixels in the mixed image
            loss_mix_trg_2d = F.kl_div(
                F.log_softmax(
                    feature_sampling(preds_2d_mix['seg_logit2'], data_batch_mix['img_indices'])[data_batch_mix['mask']],
                    dim=1),
                F.softmax(data_batch_mix['seg_logit_3d'].detach(), dim=1),
                reduction='none').sum(1).mean()
            src_label = feature_sampling(data_batch_mix['seg_label_2d'].unsqueeze(1),data_batch_mix['img_indices']).squeeze()
            prototype, mask = get_feature_(seg_logit_3d, pl_2d_, src_label[~data_batch_mix['mask']].cpu().numpy(), cfg)
            pixel_prototype = prototype[src_label[~data_batch_mix['mask']][mask].long()]
            # for source pixels in the mixed image
            loss_mix_src_2d = F.kl_div(
                F.log_softmax((feature_sampling(preds_2d_mix['seg_logit2'], data_batch_mix['img_indices'])[~data_batch_mix['mask']][mask]), dim=1),
                F.softmax(pixel_prototype.detach(), dim=1),
                reduction='none').sum(1).mean()

            loss_mix_2d = loss_mix_trg_2d+loss_mix_src_2d
            loss_mix_3d = F.cross_entropy(preds_3d_mix['seg_logit'],data_batch_mix['pl'].detach())
            loss_2d = cfg.TRAIN.XMUDA.lambda_mix_2d*loss_mix_2d
            loss_3d = cfg.TRAIN.XMUDA.lambda_mix_3d*loss_mix_3d
            sum_item['loss_mix_trg_2d'] = loss_mix_trg_2d
            sum_item['loss_mix_src_2d'] = loss_mix_src_2d
            sum_item['loss_mix_3d'] = loss_mix_3d
            del data_batch_src, data_batch_trg, data_batch_mix, preds_2d_mix,preds_3d_mix
            loss_2d.backward()
            loss_3d.backward()
        optimizer_2d.step()
        optimizer_3d.step()
        with torch.no_grad():
            for param_q, param_k in zip(model_3d.parameters(), model_3d_ema.parameters()):
                param_k.data = param_k.data.clone() * 0.99 + param_q.data.clone() * (1. - 0.99)
            for buffer_q, buffer_k in zip(model_3d.buffers(), model_3d_ema.buffers()):
                buffer_k.data = buffer_q.data.clone()
        if cur_iter == 1 or (cur_iter%sum_period == 0 and sum_period>0):
            logger.info("iters:{}".format(cur_iter) + ''.join(
                (' ' + name + ':{:.4f}').format(value.item()) for name, value in sum_item.items()))
        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            setup_validate()
            sum_item_val = {}
            class_miou_2d, class_miou_3d, class_miou_avg = validate(model_2d, model_3d, val_dataloader, cfg)
            sum_item_val['miou_2d'] = round(np.nanmean(class_miou_2d) * 100, 2)
            sum_item_val['miou_3d'] = round(np.nanmean(class_miou_3d) * 100, 2)
            sum_item_val['miou_avg'] = round(np.nanmean(class_miou_avg) * 100, 2)
            sum_item_val['class_iou_2d'] = class_miou_2d
            sum_item_val['class_iou_3d'] = class_miou_3d
            sum_item_val['class_iou_avg'] = class_miou_avg
            logger.info(''.join(
                (name + ':{}' + ' ').format(value) for name, value in sum_item_val.items()))

            setup_train()

        # ---------------------------------------------------------------------------- #
        scheduler_2d.step()
        scheduler_3d.step()

        if (ckpt > 0 and cur_iter % ckpt == 0) or cur_iter == max_iteration:
            filepath_2d = os.path.join(output_dir, 'model_2d_{}.pth'.format(cur_iter))
            filepath_3d = os.path.join(output_dir, 'model_3d_{}.pth'.format(cur_iter))
            torch.save(model_2d.state_dict(), filepath_2d)
            torch.save(model_3d.state_dict(), filepath_3d)
            print('save model_2d_{}'.format(cur_iter))
            print('save model_3d_{}'.format(cur_iter))
        gc.collect()
        torch.cuda.empty_cache()
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

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from common.config import purge_cfg
    from config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)
    logger = setup_logger('CoMoDaL', output_dir, comment='train_CoMoDaL.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))
    # check that 2D and 3D model use either both single head or both dual head
    #assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    train(cfg, output_dir)


if __name__ == '__main__':
    main()