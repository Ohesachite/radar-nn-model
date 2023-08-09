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

import models.loss as losses

from datasets.radar import Radar
from datasets.radarOut import RadarOutput
import models.radar as Models

def initialize_output(model, dataloader, device, mode):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    confusion = utils.ConfusionMatrix(dataloader.dataset.num_classes, device)
    header = 'Initialization [mode={}]:'.format(mode)
    video_prob = {}
    video_label = {}
    with torch.no_grad():
        for clip, features, target, video_idx, vid_samp_n, vid_n_samp, _, _ in metric_logger.log_every(dataloader, 100, header):
            clip = clip.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output, _, _ = model(clip, features)

            acc1, acc3 = utils.accuracy(output, target, topk=(1, 3))
            prob = F.softmax(input=output, dim=1)

            confusion.add_accuracy_by_class(target, output)

            # entire video loss
            batch_size = clip.shape[0]
            target = target.cpu().numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            vid_samp_n = vid_samp_n.cpu().numpy()
            vid_n_samp = vid_n_samp.cpu().numpy()
            for i in range(0, batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx][vid_samp_n[i], :] += prob[i]
                else:
                    video_prob[idx] = np.zeros((vid_n_samp[i], dataloader.dataset.num_classes))
                    video_prob[idx][vid_samp_n[i], :] += prob[i]
                    video_label[idx] = target[i]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@3 {top3.global_avg:.3f}'.format(top1=metric_logger.acc1, top3=metric_logger.acc3))

    confusion_matrix = confusion.get_confusion_matrix().cpu().numpy()
    print('Confusion Matrix:')
    with np.printoptions(precision=3):
        print(confusion_matrix)

    # Subsample level prediction
    return video_prob, video_label, metric_logger.acc1.global_avg

def train_one_epoch(model, criterion, optimizer, dataloader, device, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Train: [{}]'.format(epoch)
    for out, target in metric_logger.log_every(dataloader, 10, header):
        out = out.to(device)
        target = target.to(device)

        combined_out = model(out)
        loss = criterion(combined_out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc, _ = utils.accuracy(combined_out, target, topk=(1,1))
        batch_size = out.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        sys.stdout.flush()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {acc.global_avg:.3f}'.format(acc=metric_logger.acc))

def evaluate(model, criterion, dataloader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    confusion = utils.ConfusionMatrix(10, device)
    header = 'Test:'
    with torch.no_grad():
        for out, target in metric_logger.log_every(dataloader, 50, header):
            out = out.to(device)
            target = target.to(device)

            combined_out = model(out)
            loss = criterion(combined_out, target)

            confusion.add_accuracy_by_class(target, combined_out)

            acc, _ = utils.accuracy(combined_out, target, topk=(1,1))
            batch_size = out.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc'].update(acc.item(), n=batch_size)
            sys.stdout.flush()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {acc.global_avg:.3f}'.format(acc=metric_logger.acc))

    confusion_matrix = confusion.get_confusion_matrix().cpu().numpy()
    print('Confusion Matrix:')
    with np.printoptions(precision=3):
        print(confusion_matrix)

    return metric_logger.acc.global_avg, confusion

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')

    parser.add_argument('--train-path', default='data/radar/train_na_big', type=str, help='training dataset')
    parser.add_argument('--test-path', default='data/radar/test_big', type=str, help='training dataset')
    parser.add_argument('--sw-model-path', default='ckpts/sw/alt_ckpt_39.pth', type=str, help='training dataset')
    parser.add_argument('--deci-model-path', default='ckpts/sg-deci/alt_ckpt_199.pth', type=str, help='training dataset')

    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--int-layers', default=0, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    parser.add_argument('--result-file', default='', type=str, help='result file')

    args = parser.parse_args()

    return args

args = parse_args()

if args.result_file:
    result_file = open(args.result_file, "w")
else:
    result_file = None

print("torch version: ", torch.__version__)
print("torchvision version: ", torchvision.__version__)

device = torch.device('cuda')

print("Loading data")

dataset_sw = Radar(root=args.train_path, frames_per_clip=24, frame_interval=1, num_points=1024, mask_split=[0.0, 0.0, 1.0], mode=6)
dataset_deci = Radar(root=args.train_path, frames_per_clip=24, frame_interval=1, num_points=1024, mask_split=[0.0, 0.0, 1.0], mode=5)
dataset_sw_test = Radar(root=args.test_path, frames_per_clip=24, frame_interval=1, num_points=1024, mask_split=[0.0, 0.0, 1.0], mode=6)
dataset_deci_test = Radar(root=args.test_path, frames_per_clip=24, frame_interval=1, num_points=1024, mask_split=[0.0, 0.0, 1.0], mode=5)

print("Creating data loaders")

dataloader_sw = torch.utils.data.DataLoader(dataset_sw, batch_size=20, num_workers=10, pin_memory=True)
dataloader_deci = torch.utils.data.DataLoader(dataset_deci, batch_size=20, num_workers=10, pin_memory=True)
dataloader_sw_test = torch.utils.data.DataLoader(dataset_sw_test, batch_size=20, num_workers=10, pin_memory=True)
dataloader_deci_test = torch.utils.data.DataLoader(dataset_deci_test, batch_size=20, num_workers=10, pin_memory=True)

print("Creating models")

model_sw = Models.RadarP4Transformer(radius=0.5, nsamples=32, spatial_stride=32,
                                    temporal_kernel_size=3, temporal_stride=2,
                                    emb_relu=False,
                                    dim=1024, depth=5, heads=8, dim_head=128,
                                    mlp_dim=2048, num_classes=dataset_sw.num_classes)

model_deci = Models.RadarP4Transformer(radius=0.5, nsamples=32, spatial_stride=32,
                                    temporal_kernel_size=3, temporal_stride=2,
                                    emb_relu=False,
                                    dim=1024, depth=5, heads=8, dim_head=128,
                                    mlp_dim=2048, num_classes=dataset_deci.num_classes)

model_combinator = Models.RadarOutputCombinator(2, n_int_layers=0)

if torch.cuda.device_count() > 1:
    model_sw = nn.DataParallel(model_sw)
    model_deci = nn.DataParallel(model_deci)
    model_combinator = nn.DataParallel(model_combinator)
model_sw = model_sw.to(device)
model_deci = model_deci.to(device)
model_combinator = model_combinator.to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.01

optimizer = torch.optim.SGD(model_combinator.parameters(), lr=lr, momentum=args.momentum)

print("Loading checkpoints")

ckpt_sw = torch.load(args.sw_model_path, map_location='cpu')
model_sw.load_state_dict(ckpt_sw['model'])

ckpt_deci = torch.load(args.deci_model_path, map_location='cpu')
model_deci.load_state_dict(ckpt_deci['model'])

print("Initializing outputs")

train_sw_out, vid_labels_train, _ = initialize_output(model_sw, dataloader_sw, device, mode=6)
test_sw_out, vid_labels_test, init_sw_acc = initialize_output(model_sw, dataloader_sw_test, device, mode=6)
train_deci_out, _, _ = initialize_output(model_deci, dataloader_deci, device, mode=5)
test_deci_out, _, init_deci_acc = initialize_output(model_deci, dataloader_deci_test, device, mode=5)

train_out = RadarOutput(vid_labels_train, train_sw_out, train_deci_out)
test_out = RadarOutput(vid_labels_test, test_sw_out, test_deci_out)

dataloader_train = torch.utils.data.DataLoader(train_out, batch_size=10, shuffle=True, num_workers=10, pin_memory=True)
dataloader_test = torch.utils.data.DataLoader(test_out, batch_size=10, num_workers=10, pin_memory=True)

print("Training combinator")

epochs = 50
for epoch in range(epochs):
    train_one_epoch(model_combinator, criterion, optimizer, dataloader_train, device, epoch)
    acc, confusion = evaluate(model_combinator, criterion, dataloader_test, device)

if result_file is not None:
    result_file.write('Initial accuracies (sw, deci):\n')
    result_file.write('({}, {})\n'.format(init_sw_acc, init_deci_acc))
    result_file.write('Final accuracies (overall, precision, recall):\n')
    result_file.write('{}\n'.format(acc))
    result_file.write('%s\n'%str(confusion.class_precisions().cpu().numpy()))
    result_file.write('%s\n'%str(confusion.class_recalls().cpu().numpy()))
    result_file.close()
