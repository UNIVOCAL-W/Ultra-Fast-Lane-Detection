import torch
import numpy as np
import cv2
import os
from model.model import parsingNet
from utils.common import merge_config
from data.dataset import LaneClsDataset
from data.constant import LindenLane_row_anchor, bismarck_row_anchor
import torchvision.transforms as transforms
import scipy.special, tqdm

def calculate_metrics(pred_cls_label, true_cls_label, num_lanes):
    """
    计算每条车道的 TP, FP, TN, FN
    """
    tp = np.zeros(num_lanes, dtype=int)
    fp = np.zeros(num_lanes, dtype=int)
    tn = np.zeros(num_lanes, dtype=int)
    fn = np.zeros(num_lanes, dtype=int)

    for lane_idx in range(num_lanes):
        pred_lane = pred_cls_label[:, lane_idx]
        true_lane = true_cls_label[:, lane_idx]

        tp[lane_idx] = np.sum((true_lane > 0) & (pred_lane > 0))
        fp[lane_idx] = np.sum((true_lane == 0) & (pred_lane > 0))
        tn[lane_idx] = np.sum((true_lane == 0) & (pred_lane == 0))
        fn[lane_idx] = np.sum((true_lane > 0) & (pred_lane == 0))

    return tp, fp, tn, fn


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()

    # 只关注 LindenLane 数据集
    assert cfg.dataset == 'LindenLane', "Only LindenLane dataset is supported."

    cls_num_per_lane = 8
    net = parsingNet(pretrained=False, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes),
                     use_aux=False).cuda()
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {k[7:] if 'module.' in k else k: v for k, v in state_dict.items()}
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    # 数据集
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = LaneClsDataset(cfg.data_root, os.path.join(cfg.data_root, 'valid_list.txt'),
                             img_transform=img_transforms, griding_num=cfg.griding_num,
                             row_anchor=LindenLane_row_anchor, num_lanes=cfg.num_lanes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # 初始化混淆矩阵
    total_tp = np.zeros(cfg.num_lanes, dtype=int)
    total_fp = np.zeros(cfg.num_lanes, dtype=int)
    total_tn = np.zeros(cfg.num_lanes, dtype=int)
    total_fn = np.zeros(cfg.num_lanes, dtype=int)

    # 遍历测试集
    for imgs, true_cls_label in loader:
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)
            out = out[0].data.cpu().numpy()
            out = out[:, ::-1, :]  # reverse direction
            prob = scipy.special.softmax(out[:-1, :, :], axis=0)
            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out = np.argmax(out, axis=0)
            loc[out == cfg.griding_num] = 0  # 无车道点为 0
            pred_cls_label = loc.astype(int)
        
        print("true_cls_label shape:", true_cls_label.shape)
        print("pred_cls_label shape:", pred_cls_label.shape)

        # 计算每张图片的 TP, FP, TN, FN
        #tp, fp, tn, fn = calculate_metrics(pred_cls_label, true_cls_label.numpy(), cfg.num_lanes)
        tp, fp, tn, fn = calculate_metrics(pred_cls_label, true_cls_label.squeeze(0).numpy(), cfg.num_lanes)
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    # 输出结果
    for lane_idx in range(cfg.num_lanes):
        P = total_tp[lane_idx] / (total_tp[lane_idx] + total_fp[lane_idx] + 1e-9)
        R = total_tp[lane_idx] / (total_tp[lane_idx] + total_fn[lane_idx] + 1e-9)
        A = (total_tp[lane_idx] + total_tn[lane_idx]) / (total_tp[lane_idx] + total_tn[lane_idx] + total_fp[lane_idx] + total_fn[lane_idx] + 1e-9)
        
        print(f"Lane {lane_idx + 1}: TP={total_tp[lane_idx]}, FP={total_fp[lane_idx]}, TN={total_tn[lane_idx]}, FN={total_fn[lane_idx]}")
        print(f"Precision = {P:.3f}, Recall = {R:.3f}, Accuracy = {A:.3f}\n")
