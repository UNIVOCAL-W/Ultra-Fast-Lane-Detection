import torch, os
import numpy as np

import torchvision.transforms as transforms
import data.mytransforms as mytransforms
from data.constant import tusimple_row_anchor, culane_row_anchor, LindenLane_row_anchor, bismarck_row_anchor
from data.dataset import LaneClsDataset, LaneTestDataset

def get_train_loader(batch_size, data_root, griding_num, dataset, use_aux, distributed, num_lanes):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((288, 800)),
        mytransforms.MaskToTensor(),
    ])
    # foto size/8
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # neu for linden map
    img_transform_linden = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    target_transform_linden = transforms.Compose([
        mytransforms.FreeScaleMask((224, 224)),
        mytransforms.MaskToTensor(),
    ])
    # foto size/8
    segment_transform_linden = transforms.Compose([
        mytransforms.FreeScaleMask((28, 28)),
        mytransforms.MaskToTensor(),
    ])

    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100), # up and down in px
        mytransforms.RandomLROffsetLABEL(200) # left and right in px
    ])
    if dataset == 'CULane':
        train_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'list/train_gt_small.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           segment_transform=segment_transform, 
                                           row_anchor = culane_row_anchor,
                                           griding_num=griding_num, use_aux=use_aux, num_lanes = num_lanes)
        cls_num_per_lane = 18

    elif dataset == 'Tusimple':
        train_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           griding_num=griding_num, 
                                           row_anchor = tusimple_row_anchor,
                                           segment_transform=segment_transform,use_aux=use_aux, num_lanes = num_lanes)
        cls_num_per_lane = 56
    elif dataset == 'LindenLane' :
        train_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_list.txt'), ## 要改
                                           img_transform=img_transform_linden, target_transform=target_transform_linden,
                                           simu_transform = simu_transform,
                                           segment_transform=segment_transform_linden, 
                                           row_anchor = LindenLane_row_anchor,
                                           griding_num=griding_num, use_aux=use_aux, num_lanes = num_lanes)
        cls_num_per_lane = 8
    elif dataset == 'bismarck' :
        train_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_list.txt'), ## 要改
                                           img_transform=img_transform_linden, target_transform=target_transform_linden,
                                           simu_transform = simu_transform,
                                           segment_transform=segment_transform_linden, 
                                           row_anchor = bismarck_row_anchor,
                                           griding_num=griding_num, use_aux=use_aux, num_lanes = num_lanes)
        cls_num_per_lane = 13
    else:
        raise NotImplementedError

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=4)

    return train_loader, cls_num_per_lane

def get_test_loader(batch_size, data_root, griding_num, dataset, use_aux, distributed, num_lanes):
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # neu for linden map
    img_transform_linden = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if dataset == 'CULane':
        test_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'list/test.txt'),
                                           img_transform=img_transform,  
                                           row_anchor = culane_row_anchor,
                                           griding_num=griding_num, use_aux=use_aux, num_lanes = num_lanes)
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        test_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'test.txt'),
                                           img_transform=img_transform,  
                                           row_anchor = tusimple_row_anchor,
                                           griding_num=griding_num, use_aux=use_aux, num_lanes = num_lanes)
        cls_num_per_lane = 56
    elif dataset == 'LindenLane':
        test_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'valid_list.txt'),
                                           img_transform=img_transform_linden,  
                                           row_anchor = LindenLane_row_anchor,
                                           griding_num=griding_num, use_aux=use_aux, num_lanes = num_lanes)
        cls_num_per_lane = 8

    if distributed:
        sampler = SeqDistributedSampler(test_dataset, shuffle = False)
    else:
        sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler, num_workers=4)
    return loader, cls_num_per_lane


class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    '''
    Change the behavior of DistributedSampler to sequential distributed sampling.
    The sequential sampling helps the stability of multi-thread testing, which needs multi-thread file io.
    Without sequentially sampling, the file io on thread may interfere other threads.
    '''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size


        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank * self.rank : num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)