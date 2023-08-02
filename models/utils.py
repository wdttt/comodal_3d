import numpy as np
import torch

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))
def feature_sampling(x,img_indices):
    img_feats = []
    for i in range(x.shape[0]):
        img_feats.append(x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
    img_feats = torch.cat(img_feats, 0)
    return img_feats
def feature_sampling_reverse(x,img_indices):
    b,c,h,w = x.shape
    mask = torch.full([b, h, w], True)
    for i in range(x.shape[0]):
        mask[i][img_indices[i][:, 0], img_indices[i][:, 1]] = False
    return mask

def get_feature_(feat,label,gt,cfg):
    N,C = feat.shape
    feat_class=torch.zeros([C,C]).cuda()
    mask_class = np.full([len(gt),],False)
    for i in range(cfg.MODEL_3D.NUM_CLASSES):
        if len(feat[label==i])>0:
            feat_class[i] = feat[label==i].mean(0)
            mask_class[gt==i] = True
    return feat_class,mask_class
def get_feature(feat,label,cfg):
    N,C = feat.shape
    feat_class=torch.zeros([10,C]).cuda()
    for i in range(cfg.MODEL_3D.NUM_CLASSES):
        if len(feat[label==i])>0:
            feat_class[i] = feat[label==i].mean(0)
    return feat_class
def get_masks(img, mask_type='class', mask_props='constant',labels=None,props=0.4):
    dev = img.device
    img_shape = img.shape
    assert mask_props in ['constant', 'random']
    if mask_type == 'class':
        return get_class_masks(labels)
    elif mask_type == 'cut':
        if mask_props == 'constant':
            cut_mask_props = props
        else:
            # cut_mask_props = np.random.beta(0.5, 0.5, img_shape[0])
            cut_mask_props = np.random.beta(5, 5, img_shape[0])
        return get_cut_masks(img_shape, mask_props=cut_mask_props, device=dev)
    else:
        raise ValueError

def get_cut_masks(img_shape, random_aspect_ratio=True, within_bounds=True, mask_props=0.4, device='cuda:0'):
    n, _, h, w = img_shape

    if random_aspect_ratio:
        y_props = np.exp(np.random.uniform(low=0.0, high=1, size=(n, 1)) * np.log(mask_props))
        x_props = mask_props / y_props

    else:
        y_props = x_props = np.sqrt(mask_props)

    sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array((h, w))[None, None, :])
    if within_bounds:
        positions = np.round(
            (np.array((h, w)) - sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(positions, positions + sizes, axis=2)
    else:
        centres = np.round(np.array((h, w)) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

    masks = []
    mask = torch.zeros((n, 1, h, w), device=device).long()
    for i, sample_rectangles in enumerate(rectangles):
        y0, x0, y1, x1 = sample_rectangles[0]
        mask[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1
        masks.append(mask[i].unsqueeze(0))
    return masks
def get_class_masks(labels, mask_wo_ignore=False, rcs_classes=None, rcs_classesprob=None, temp=0.01):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        if rcs_classes is None:
            class_choice = np.random.choice(
                nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
            classes = classes[torch.Tensor(class_choice).long()]
            if mask_wo_ignore:
                classes = classes[classes != 255]
        else:
            num_classes2choice = int((nclasses + nclasses % 2) / 2)
            classes = classes[classes != 255]
            new_classes = []
            new_classesprob = []

            for i, c in enumerate(rcs_classes):
                if c in list(classes.cpu().numpy()):
                    new_classes.append(c)
                    new_classesprob.append(rcs_classesprob[i])
            new_classesprob = torch.tensor(list(new_classesprob))
            new_classesprob = torch.softmax(new_classesprob / temp, dim=-1)
            class_choice = np.random.choice(new_classes, num_classes2choice, p=new_classesprob.numpy(), replace=False)
            classes = torch.Tensor(class_choice).long().cuda()

        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks
def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask