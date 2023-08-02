import torch
from functools import partial


def collate_scn_base(input_dict_list):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :return: Collated data batch as dict
    """
    locs=[]
    feats=[]
    orig_locs = []
    orig_feats = []
    orig_seg_label = []
    labels=[]
    orig_points = []
    imgs = []
    img_idxs = []
    label_dense=[]
    orig_label_dense = []
    orig_imgs = []
    orig_img_idxs = []

    for idx, input_dict in enumerate(input_dict_list):
        if 'coords' in input_dict.keys():
            coords = torch.from_numpy(input_dict['coords'])
            batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
            locs.append(torch.cat([coords, batch_idxs], 1))
            feats.append(torch.from_numpy(input_dict['feats']))
        if 'orig_coords' in input_dict.keys():
            orig_coords = torch.from_numpy(input_dict['orig_coords'])
            orig_batch_idxs = torch.LongTensor(orig_coords.shape[0], 1).fill_(idx)
            orig_locs.append(torch.cat([orig_coords, orig_batch_idxs], 1))
            orig_feats.append(torch.from_numpy(input_dict['orig_feats']))
        if 'seg_label' in input_dict.keys():
            labels.append(torch.from_numpy(input_dict['seg_label']))
        if 'orig_points' in input_dict.keys():
            orig_points.append(input_dict['orig_points'])
        if 'orig_seg_label' in input_dict.keys():
            orig_seg_label.append(torch.from_numpy(input_dict['orig_seg_label']))
        if 'img' in input_dict.keys():
            imgs.append(torch.from_numpy(input_dict['img']))
        if 'label_dense' in input_dict.keys():
            label_dense.append(torch.from_numpy(input_dict['label_dense']))
        if 'img_indices' in input_dict.keys():
            img_idxs.append(input_dict['img_indices'])
        if 'orig_img' in input_dict.keys():
            orig_imgs.append(torch.from_numpy(input_dict['orig_img']))
        if 'orig_img_indices' in input_dict.keys():
            orig_img_idxs.append(input_dict['orig_img_indices'])
        if 'orig_label_dense' in input_dict.keys():
            orig_label_dense.append(torch.from_numpy(input_dict['orig_label_dense']))



    out_dict ={}
    if locs and feats:
        locs = torch.cat(locs, 0)
        feats = torch.cat(feats, 0)
        out_dict['x']= [locs, feats]
    if orig_points:
        out_dict['orig_points'] = orig_points
    if labels:
        labels = torch.cat(labels, 0)
        out_dict['seg_label'] = labels
    if orig_seg_label:
        out_dict['orig_seg_label'] = torch.cat(orig_seg_label, dim=0)
    if imgs:
        out_dict['img'] = torch.stack(imgs)
    if label_dense:
        out_dict['label_dense'] = torch.stack(label_dense)
    if img_idxs:
        out_dict['img_indices'] = img_idxs
    if orig_imgs:
        out_dict['orig_img'] = torch.stack(orig_imgs)
    if orig_img_idxs:
        out_dict['orig_img_indices'] = orig_img_idxs
    if orig_label_dense:
        out_dict['orig_label_dense'] =  torch.stack(orig_label_dense)

    if orig_locs and orig_feats:
        orig_locs = torch.cat(orig_locs, 0)
        orig_feats = torch.cat(orig_feats, 0)
        out_dict['orig_x'] = [orig_locs, orig_feats]

    return out_dict


def get_collate_scn(is_train):
    return partial(collate_scn_base,
                   )
