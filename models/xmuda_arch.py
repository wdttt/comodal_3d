import os
import torch
import torch.nn as nn
from models.resnet34_unet import UNetResNet34
from models.scn_unet import UNetSCN


class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 backbone_2d_kwargs,
                 ):
        super(Net2DSeg, self).__init__()
        self.backbone_2d = backbone_2d
        # 2D image network
        if backbone_2d == 'UNetResNet34':
            self.net_2d = UNetResNet34(**backbone_2d_kwargs)
            feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        # segmentation head
        self.conv = nn.Conv2d(feat_channels, num_classes,kernel_size=1)
        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.conv2 = nn.Conv2d(feat_channels, num_classes,kernel_size=1)

    def forward(self, data_batch):
        # (batch_size, 3, H, W)
        img = data_batch['img']
        seg= self.net_2d(img)
        out_seg = self.conv(seg)

        preds = {
            'feats': seg,
            'seg_logit': out_seg,
        }
        if self.dual_head:
            preds['seg_logit2'] = self.conv2(seg)
        return preds


class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 ):
        super(Net3DSeg, self).__init__()

        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = UNetSCN(**backbone_3d_kwargs)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # segmentation head
        self.linear = nn.Linear(self.net_3d.out_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.net_3d.out_channels, num_classes)
    def forward(self, data_batch):
        #feats,feats_ = self.net_3d(data_batch['x'])
        feats= self.net_3d(data_batch['x'])
        x = self.linear(feats)

        preds = {
            'feats': feats,
            #'feats_': feats_,
            'seg_logit': x,
        }

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(feats)
        return preds
def test_Net2DSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      backbone_2d='UNetResNet34',
                      backbone_2d_kwargs={},
                      dual_head=True)

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)


def test_Net3DSeg():
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)

    feats = feats.cuda()

    net_3d = Net3DSeg(num_seg_classes,
                      dual_head=True,
                      backbone_3d='SCN',
                      backbone_3d_kwargs={'in_channels': in_channels})

    net_3d.cuda()
    out_dict = net_3d({
        'x': [coords, feats],
    })
    for k, v in out_dict.items():
        print('Net3DSeg:', k, v.shape)
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    test_Net2DSeg()

