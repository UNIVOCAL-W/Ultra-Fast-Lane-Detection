import torch, os, datetime
import numpy as np

from model.model import parsingNet
from data.dataloader import get_train_loader, get_valid_loader

from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time

import matplotlib.pyplot as plt

all_epoch_loss_train = []
all_epoch_loss_valid = []

acc_top1_train = []
acc_top2_train = []
acc_top3_train = []
acc_top1_valid = []
acc_top2_valid = []
acc_top3_valid = []


# cd C:\BA_Workspace\Ultra-Fast-Lane-Detection
# C:\Users\13208\AppData\Local\Programs\Python\Python312\python.exe train.py configs/culane.py
def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cuda(), cls_label.long().cuda(), seg_label.long().cuda()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}
    else:
        img, cls_label = data_label
        img, cls_label = img.cuda(), cls_label.long().cuda()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results, logger, global_step):
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        # 判断 global_step 是否为 None
        if global_step is not None and global_step % 20 == 0:
            logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)

        # if global_step % 20 == 0:
        #     logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss


def train(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, use_aux):
    net.train()
    epoch_loss = []
    acc_top1 = []
    acc_top2 = []
    acc_top3 = []
    progress_bar = dist_tqdm(train_loader) 
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx
        

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux)

        loss = calc_loss(loss_dict, results, logger, global_step)
        epoch_loss.append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)

        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            for name, value in kwargs.items():
                if name == 'top1':
                    acc_top1.append(float(value))
                elif name == 'top2':
                    acc_top2.append(float(value))
                elif name == 'top3':
                    acc_top3.append(float(value))

            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
        t_data_0 = time.time()


    
    avg_loss = sum(epoch_loss) / len(epoch_loss)
    all_epoch_loss_train.append(float(avg_loss))

    avg_top1 = sum(acc_top1) / len(acc_top1)
    acc_top1_train.append(float(avg_top1))

    avg_top2 = sum(acc_top2) / len(acc_top2)
    acc_top2_train.append(float(avg_top2))

    avg_top3 = sum(acc_top3) / len(acc_top3)
    acc_top3_train.append(float(avg_top3))



def validate(net, data_loader, loss_dict, logger, epoch, metric_dict, use_aux):
    net.eval()
    #reset_metrics(metric_dict)  # 清空之前的metric统计
    val_losses = []

    acc_top1 = []
    acc_top2 = []
    acc_top3 = []

    for b_idx, data_label in enumerate(data_loader):
        
        reset_metrics(metric_dict)  # 清空之前的metric统计

        results = inference(net, data_label, use_aux)
        
        loss = calc_loss(loss_dict, results, logger, global_step=None)  
        val_losses.append(float(loss))

        results = resolve_val_data(results, use_aux)
        update_metrics(metric_dict, results)

        for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
            val_metric = me_op.get()
            logger.add_scalar('val_metric/' + me_name, val_metric, epoch)
            #dist_print(f"[Validation] {me_name}: {val_metric:.4f}")
            if me_name == 'top1':
                acc_top1.append(float(val_metric))
            elif me_name == 'top2':
                acc_top2.append(float(val_metric))
            elif me_name == 'top3':
                acc_top3.append(float(val_metric))

    avg_val_loss = sum(val_losses) / len(val_losses) #if len(val_losses) > 0 else 0.0
    avg_top1 = sum(acc_top1) / len(acc_top1)
    avg_top2 = sum(acc_top2) / len(acc_top2)
    avg_top3 = sum(acc_top3) / len(acc_top3)

    dist_print(f"[Validation] Epoch={epoch}, val_loss={avg_val_loss:.4f}")
    dist_print(f"[Validation] top1 : {avg_top1:.4f}")
    dist_print(f"[Validation] top2 : {avg_top2:.4f}")
    dist_print(f"[Validation] top3 : {avg_top3:.4f}")
    logger.add_scalar('val/loss', avg_val_loss, epoch)

    # for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
    #     val_metric = me_op.get()
    #     logger.add_scalar('val_metric/' + me_name, val_metric, epoch)
    #     dist_print(f"[Validation] {me_name}: {val_metric:.4f}")
    #     if me_name == 'top1':
    #         acc_top1_valid.append(float(val_metric))
    #     elif me_name == 'top2':
    #         acc_top2_valid.append(float(val_metric))
    #     elif me_name == 'top3':
    #         acc_top3_valid.append(float(val_metric))
    

    net.train()  # 验证结束后记得切回 train 模式

    
    all_epoch_loss_valid.append(float(avg_val_loss))

    
    acc_top1_valid.append(float(avg_top1))
    acc_top2_valid.append(float(avg_top2))
    acc_top3_valid.append(float(avg_top3))

    



def plot_loss_curve(train_losses, val_losses):
    epochs = range(len(train_losses))  

    plt.figure(figsize=(8, 6))
    # 绘制训练集 Loss 曲线
    plt.plot(epochs, train_losses, marker='o', color='blue', label='Train Loss')

    # 绘制验证集 Loss 曲线
    plt.plot(epochs, val_losses, marker='x', color='red', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('all_epochs_loss_curve.png')
    plt.show()

def plot_top1_curve(train_top1, val_top1):
    epochs = range(len(train_top1))  

    plt.figure(figsize=(8, 6))
    
    plt.plot(epochs, train_top1, marker='o', color='blue', label='Train Top1')

    plt.plot(epochs, val_top1, marker='x', color='red', label='Validation Top1')
    plt.xlabel('Epoch')
    plt.ylabel('Average Top1')
    plt.title('Average Top1 per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('all_epochs_top1_curve.png')
    plt.show()

def plot_top2_curve(train_top2, val_top2):
    epochs = range(len(train_top2))  

    plt.figure(figsize=(8, 6))
    
    plt.plot(epochs, train_top2, marker='o', color='blue', label='Train Top2')

    plt.plot(epochs, val_top2, marker='x', color='red', label='Validation Top2')
    plt.xlabel('Epoch')
    plt.ylabel('Average Top2')
    plt.title('Average Top2 per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('all_epochs_top2_curve.png')
    plt.show()

def plot_top3_curve(train_top3, val_top3):
    epochs = range(len(train_top3))  

    plt.figure(figsize=(8, 6))
    
    plt.plot(epochs, train_top3, marker='o', color='blue', label='Train Top3')

    plt.plot(epochs, val_top3, marker='x', color='red', label='Validation Top3')
    plt.xlabel('Epoch')
    plt.ylabel('Average Top3')
    plt.title('Average Top3 per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('all_epochs_top3_curve.png')
    plt.show()




if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']


    train_loader, cls_num_per_lane = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, cfg.dataset, cfg.use_aux, distributed, cfg.num_lanes)

    valid_loader, cls_num_per_lane = get_valid_loader(cfg.batch_size * 2, cfg.data_root, cfg.griding_num, cfg.dataset, cfg.use_aux, distributed, cfg.num_lanes)

    net = parsingNet(pretrained = True, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),use_aux=cfg.use_aux).cuda()

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0



    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    cp_projects(args.auto_backup, work_dir)

    for epoch in range(resume_epoch, cfg.epoch):

        train(net, train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg.use_aux)
        
        save_model(net, optimizer, epoch ,work_dir, distributed)
        validate(net, valid_loader, loss_dict, logger, epoch, metric_dict, cfg.use_aux)
    plot_loss_curve(all_epoch_loss_train, all_epoch_loss_valid)
    plot_top1_curve(acc_top1_train, acc_top1_valid)
    plot_top2_curve(acc_top2_train, acc_top2_valid)
    plot_top3_curve(acc_top3_train, acc_top3_valid)
    logger.close()
