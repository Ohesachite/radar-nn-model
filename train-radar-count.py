from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import utils

from scheduler import WarmupMultiStepLR
import models.loss as losses

from datasets.radar import Radar
import models.radar as Models

def train_one_epoch(model, criterion, optimizer, lr, data_loader, device, epoch, batch_size, within_period_thres=0.5, clip_size=64, output_file=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    within_period_loss_func = nn.BCELoss()
    count_loss_func = nn.L1Loss()

    header = 'Train: [{}]'.format(epoch)
    for video, vid_features, targets in metric_logger.log_every(data_loader, 100, header):
        start_time = time.time()
        video, vid_features = video.to(device), vid_features.to(device)
        video = torch.squeeze(video, 0)
        vid_features = torch.squeeze(video, 0)
        vid_len = video.shape[0]

        y1 = targets[0].to(device)
        y1 = torch.squeeze(y1, 0)
        y2 = targets[1].to(device)
        y2 = torch.squeeze(y2, 0)
        count_target = targets[4].to(device)

        optimal_stride = targets[3].item()
        num_batches = int(np.ceil(vid_len/clip_size/optimal_stride/batch_size))
        total_vid_loss = 0.
        mean_period_loss = 0.
        mean_within_period_loss = 0.
        period_list = []
        within_period_list = []
        for batch_idx in range(num_batches):
            idxes = torch.arange(start=batch_idx*batch_size*clip_size*optimal_stride,
                                end=(batch_idx+1)*batch_size*clip_size*optimal_stride,
                                step=optimal_stride, device=device, dtype=torch.int64)
            idxes = torch.clip(idxes, min=0, max=vid_len-1)
            curr_frames = video[idxes,:,:]
            curr_frames = torch.reshape(curr_frames, (batch_size, clip_size, -1, 3))
            curr_features = vid_features[idxes,:,:]
            curr_features = torch.reshape(curr_features, (batch_size, clip_size, 2, -1))

            period_out, within_period_out = model(curr_frames, curr_features)

            period_target = y1[idxes,:]
            period_target = torch.reshape(period_target, (batch_size, clip_size, -1))
            within_period_target = y2[idxes]
            within_period_target = torch.reshape(within_period_target, (batch_size, clip_size))
            
            period_loss = criterion(period_out.transpose(1,2), period_target.transpose(1,2))
            within_period_loss = within_period_loss_func(within_period_out, within_period_target)

            mean_period_loss += period_loss
            mean_within_period_loss += within_period_loss

            flat_period = torch.flatten(period_out, end_dim=1)
            flat_within_period = torch.flatten(within_period_out, end_dim=1)

            period_list.append(flat_period)
            within_period_list.append(flat_within_period)

        mean_period_loss = mean_period_loss / num_batches
        mean_within_period_loss = mean_within_period_loss / num_batches

        total_vid_loss = total_vid_loss + mean_period_loss + batch_size * mean_within_period_loss

        period_all = torch.repeat_interleave(torch.cat(period_list), optimal_stride, dim=0)[:vid_len, :]
        within_period_all = torch.repeat_interleave(torch.cat(within_period_list), optimal_stride)[:vid_len]

        count_pred = torch.reciprocal(torch.add(torch.argmax(period_all, dim=1), 1) * optimal_stride)
        count_pred = torch.where(torch.logical_and(within_period_all > within_period_thres, count_pred < (1.0 / optimal_stride)), count_pred, 0.)
        count_pred = torch.round(torch.unsqueeze(torch.sum(count_pred), 0))

        count_loss = count_loss_func(count_pred, count_target)
        if count_target.item() != 0:
            norm_count_loss = count_loss.item() / count_target.item()
        else:
            if count_loss.item() == 0:
                norm_count_loss = 0.0
            else:
                norm_count_loss = 1.0
        total_vid_loss = total_vid_loss + count_loss

        optimizer.zero_grad()
        total_vid_loss.backward()
        optimizer.step()

        metric_logger.update(total_vid_loss=total_vid_loss.item(), count_error=count_loss.item(), norm_count_error=norm_count_loss, period_loss=mean_period_loss, within_period_loss=mean_within_period_loss)
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        sys.stdout.flush()

    if output_file is not None:
        output_file.write('Training at Epoch {}:\n'.format(epoch))
        output_file.write('Total_vid_loss: {tvl.global_avg:.3f}, Count MAE: {ce.global_avg:.3f}, Normalized Count MAE: {nce.global_avg:.3f}, Period Loss: {mpl.global_avg:.3f}, Within Period Loss: {mwpl.global_avg:.3f}\n'.format(tvl=metric_logger.total_vid_loss, ce=metric_logger.count_error, nce=metric_logger.norm_count_error, mpl=metric_logger.period_loss, mwpl=metric_logger.within_period_loss))
        output_file.flush()

def evaluate(model, data_loader, device, batch_size, strides=[1,2], within_period_thres=0.5, clip_size=64, output_file=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    avg_period_mae_func = nn.L1Loss()
    count_mae_func = nn.L1Loss()

    header = 'Test:'
    with torch.no_grad():
        for video, vid_features, targets in metric_logger.log_every(data_loader, 10, header):
            video = video.to(device, non_blocking=True)
            vid_features = vid_features.to(device, non_blocking=True)
            video = torch.squeeze(video, 0)
            vid_features = torch.squeeze(video, 0)
            vid_len = video.shape[0]
            avg_period_target = torch.mul(targets[2], targets[3])
            avg_period_target = avg_period_target.to(device)
            count_target = targets[4].to(device)

            periods_list = []
            scores = []
            within_periods_list = []

            for stride in strides:
                num_batches = int(np.ceil(vid_len/clip_size/stride/batch_size))
                period_stride_list = []
                within_period_stride_list = []
                for batch_idx in range(num_batches):
                    idxes = torch.arange(start=batch_idx*batch_size*clip_size*stride, 
                                        end=(batch_idx+1)*batch_size*clip_size*stride, 
                                        step=stride, device=device, dtype=torch.int64)
                    idxes = torch.clip(idxes, min=0, max=vid_len-1)
                    curr_frames = video[idxes,:,:]
                    curr_frames = torch.reshape(curr_frames, (batch_size, clip_size, -1, 3))
                    curr_features = vid_features[idxes,:,:]
                    curr_features = torch.reshape(curr_features, (batch_size, clip_size, 2, -1))

                    period_out, within_period_out = model(curr_frames, curr_features)

                    flat_period = torch.flatten(period_out, end_dim=1)
                    flat_within_period = torch.flatten(within_period_out, end_dim=1)

                    period_stride_list.append(flat_period)
                    within_period_stride_list.append(flat_within_period)

                period_stride = torch.cat(period_stride_list)
                per_frame_periods = torch.add(torch.argmax(period_stride, dim=1), 1)
                within_period_stride = torch.cat(within_period_stride_list)

                periods_list.append(period_stride)
                within_periods_list.append(within_period_stride)

                pred_period_conf = torch.max(F.softmax(period_stride, dim=1), 1)[0]
                pred_period_conf = torch.where(per_frame_periods < 3, 0., pred_period_conf)
                within_period_score = within_period_stride * pred_period_conf
                pred_score = torch.mean(torch.sqrt(within_period_score))
                scores.append(pred_score)

            argmax_strides = np.argmax(scores)
            chosen_stride = strides[argmax_strides]

            period_all = torch.repeat_interleave(periods_list[argmax_strides], chosen_stride, dim=0)[:vid_len,:]
            within_period_all = torch.repeat_interleave(within_periods_list[argmax_strides], chosen_stride)[:vid_len]

            count_pred = torch.reciprocal(torch.add(torch.argmax(period_all, dim=1), 1) * chosen_stride)
            within_period_binary = torch.logical_and(within_period_all > within_period_thres, count_pred < (1.0 / chosen_stride))
            periodic_idxes = within_period_binary.nonzero()

            count_pred = torch.where(within_period_binary, count_pred, 0.)
            if periodic_idxes.shape[0] > 0:
                avg_period_pred = torch.reciprocal(torch.unsqueeze(torch.mean(count_pred[periodic_idxes]), 0))
            else:
                avg_period_pred = torch.tensor([1.0], device=device)
            count_pred = torch.round(torch.unsqueeze(torch.sum(count_pred), 0))

            avg_period_mae = avg_period_mae_func(avg_period_pred, avg_period_target)
            norm_avg_period_mae = avg_period_mae.item() / avg_period_target.item()
            count_mae = count_mae_func(count_pred, count_target)
            if count_target.item() != 0:
                norm_count_mae = count_mae.item() / count_target.item()
            else:
                if count_mae.item() == 0:
                    norm_count_mae = 0.0
                else:
                    norm_count_mae = 1.0

            metric_logger.update(avg_period_mae=avg_period_mae.item(), norm_avg_period_mae=norm_avg_period_mae, count_error=count_mae.item(), norm_count_mae=norm_count_mae)

    if output_file is not None:
        output_file.write('Subsequent Testing:\n')
        output_file.write('Average Period MAE: {apm.global_avg:.3f}, Normalized Average Period MAE: {napm.global_avg:.3f}, Count MAE: {ce.global_avg:.3f}, Normalized Count MAE: {nce.global_avg:.3f}\n'.format(apm=metric_logger.avg_period_mae, napm=metric_logger.norm_avg_period_mae, ce=metric_logger.count_error, nce=metric_logger.norm_count_mae))
        output_file.flush()

def main():
    training_batch_size = 5
    testing_batch_size = 10
    workers = 10

    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    print("Loading data")

    dataset_train = Radar(
            root='/home/alan/Documents/radar-nn-model/data/radar/train',
            num_points=1024,
            frames_per_clip=64,
            mode=2,
            mask_split=[1.0, 0.0, 0.0]
    )

    dataset_test = Radar(
            root='/home/alan/Documents/radar-nn-model/data/radar/test',
            num_points=1024,
            frames_per_clip=64,
            mode=1,
            mask_split=[0.0, 0.0, 1.0]
    )

    print("Creating data loaders")

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=workers, pin_memory=True)
    
    print("Creating model")

    clip_size=64
    model = Models.RadarPeriodEstimator(radius=0.5, nsamples=32, embedding_dim=512, n_frames=clip_size)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    load = True
    if load:
        print("Loading checkpoint")
        checkpoint = torch.load(os.path.join('ckpts', 'rep_counting_ckpt_50.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    output_to_file = True
    output_file_name = 'results/result.out'
    if output_to_file:
        output_file = open(output_file_name, "w")
    else:
        output_file = None

    print("Start training")
    start_time = time.time()
    for epoch in range(51, 100):

        train_one_epoch(model, criterion, optimizer, lr, data_loader_train, device, epoch, training_batch_size, clip_size=clip_size, output_file=output_file)

        evaluate(model, data_loader_test, device, testing_batch_size, clip_size=clip_size, output_file=output_file)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr,
            'epoch': epoch}
        utils.save_on_master(
            checkpoint,
            os.path.join('ckpts', 'rep_counting_ckpt_{}.pth'.format(epoch)))

if __name__ == "__main__":
    main()
