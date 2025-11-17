r""" Visual Prompt Encoder training (validation) code """
import os
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

from model.VRP_encoder import VRP_encoder
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred


def eval(args, model, sam_model, dataloader, training):
    r""" Train VRP_encoder model """

    utils.fix_randseed(0) 
    model.eval()
    average_meter = AverageMeter(dataloader.dataset)

    from time import time
    t_el=0
    for idx, batch in enumerate(dataloader):
        
        batch = utils.to_cuda(batch)
        s =time()
        protos, _ = model(args.condition, batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), training)

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        logit_mask = low_masks
        t_el += time() -s
        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        # print(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.update(area_inter, area_union, batch['class_id'], None)
        average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=100)
        if (idx+1) % 50 == 0:
            lat = t_el / (idx*2)
            fps = 1/lat
            print(f'fps: {fps:.1f}')
    average_meter.write_result('Validation', 0)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/scratch/gtrivigno/datasets')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss', 'lvis', 'paco_part', 'fss', 'pascal_part'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--sam_version', type=str, default='vit_h')
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--condition', type=str, default='mask', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None)    
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    args = parser.parse_args()
    # Distributed setting
    device = torch.device('cuda')
    Logger.initialize(args, training=False)
    utils.fix_randseed(args.seed)
    # Model initialization
    model = VRP_encoder(args, args.backbone, False)
    if args.resume is not None:
        ck = torch.load(args.resume)
        ck = {k.replace('module.', ''):v for k, v in ck.items()}
        model.load_state_dict(ck)   
    Logger.log_params(model)

    sam_model = SAM_pred(args.sam_version)
    sam_model.to(device)
    model.to(device)
    
    Evaluator.initialize(args)

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')
    with torch.no_grad():
        val_loss, val_miou, val_fb_iou = eval(args, model, sam_model, dataloader_val, training=False)
    print(val_miou)