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
import json

def initialize_output(model, dataloader, device, mode):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    confusion = utils.ConfusionMatrix(dataloader.dataset.num_classes, device)
    header = 'Initialization [mode={}]:'.format(mode)
    video_prob = {}
    clip_prob = {}
    video_label = {}
    with torch.no_grad():
        for clip, features, target, video_idx, vid_samp_n, vid_n_samp, _, _, clip_num, num_clips in metric_logger.log_every(dataloader, 100, header):
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
            clip_num = clip_num.cpu().numpy()
            num_clips = num_clips.cpu().numpy()
            for i in range(0, batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx][vid_samp_n[i], :] += prob[i]
                else:
                    video_prob[idx] = np.zeros((vid_n_samp[i], dataloader.dataset.num_classes))
                    video_prob[idx][vid_samp_n[i], :] += prob[i]
                    video_label[idx] = target[i]

                if idx in clip_prob:
                    clip_prob[idx][clip_num[i], :] += prob[i]
                else:
                    clip_prob[idx] = np.zeros((num_clips[i], dataloader.dataset.num_classes))
                    clip_prob[idx][clip_num[i], :] += prob[i]
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

    seg_pred = {k: np.argmax(v, axis=1) for k, v in video_prob.items()}
    predseg_correct = [seg_pred[k][i] == video_label[k] for k in seg_pred for i in range(len(seg_pred[k]))]
    seg_acc = np.mean(predseg_correct)

    class_count = [0] * dataloader.dataset.num_classes
    class_correct = [0] * dataloader.dataset.num_classes

    for k, v in seg_pred.items():
        label = video_label[k]
        for v2 in v:
            class_count[label] += 1
            class_correct[label] += (v2==label)
    class_acc = [c/float(s) for c, s in zip(class_correct, class_count)]

    print(' * Segmented Clip Acc@1 %f'%seg_acc)
    print(' * Segment Clip Acc by Class@1 %s'%str(class_acc))

    # Subsample level prediction
    return video_prob, clip_prob, video_label, seg_acc, class_acc

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

# Assume dataset is radarOut
def fusion_evaluate(dataset1, dataset2, clipout1, clipout2, threshold=0.68):
    fuse_preds = {}
    fuse1_preds = {}
    fuse2_preds = {}
    class_counts = [0] * 10
    class_correct = [0] * 10
    for k in dataset1.probs.keys():
        preds1 = np.argmax(dataset1.probs[k], axis=1)
        preds2 = np.argmax(dataset2.probs[k], axis=1)

        fuse1_preds[k] = preds1
        fuse2_preds[k] = preds2

        window = 11
        selection = 1
        for clipi in range(min(clipout1[k].shape[0], clipout2[k].shape[0]) - (window - 1)):
            window1_prob = np.argmax(clipout1[k][:window], axis=1)
            window2_prob = np.argmax(clipout2[k][:window], axis=1)

            window1_pred = np.bincount(window1_prob, minlength=10)
            window2_pred = np.bincount(window2_prob, minlength=10)

            window1_modes = np.argwhere(window1_pred == np.amax(window1_pred))
            window2_modes = np.argwhere(window2_pred == np.amax(window2_pred))

            if len(window1_modes) == 1 and len(window2_modes == 1):
                if window1_modes[0] == window2_modes[0]:
                    if window1_modes[0] == 2 or window1_modes[0] == 3:
                        selection = 0
                    break

        if selection == 0:
            fuse_preds[k] = preds1
        else:
            fuse_preds[k] = preds2

    acc = np.mean([fuse_preds[k][i]==dataset1.labels[k] for k in fuse_preds for i in range(len(fuse_preds[k]))])

    for k, v in fuse_preds.items():
        label = dataset1.labels[k]
        for v2 in v:
            class_counts[label] += 1
            class_correct[label] += (v2==label)
    class_acc = [c/float(s) for c, s in zip(class_correct, class_counts)]

    return fuse_preds, fuse1_preds, fuse2_preds, acc, class_acc

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')

    parser.add_argument('--train-path', default='data/radar/train_na_old', type=str, help='training dataset')
    parser.add_argument('--test-path', default='data/radar/test_old', type=str, help='training dataset')
    parser.add_argument('--sd-model-path', default='ckpts/sw/alt_ckpt_39.pth', type=str, help='training dataset')
    parser.add_argument('--bd-model-path', default='ckpts/sw/alt_ckpt_39.pth', type=str, help='training dataset')

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

# dataset_sd = Radar(root=args.train_path, frames_per_clip=24, frame_interval=1, num_points=1024, mask_split=[0.0, 0.0, 1.0], mode=6, decay=0.002, retclipnum=True)
# dataset_bd = Radar(root=args.train_path, frames_per_clip=24, frame_interval=1, num_points=1024, mask_split=[0.0, 0.0, 1.0], mode=6, decay=0.02, retclipnum=True)
dataset_sd_test = Radar(root=args.test_path, frames_per_clip=24, frame_interval=1, num_points=1024, mask_split=[0.0, 0.0, 1.0], mode=6, decay=0.002, retclipnum=True)
dataset_bd_test = Radar(root=args.test_path, frames_per_clip=24, frame_interval=1, num_points=1024, mask_split=[0.0, 0.0, 1.0], mode=6, decay=0.02, retclipnum=True)

print("Creating data loaders")

# dataloader_sd = torch.utils.data.DataLoader(dataset_sd, batch_size=20, num_workers=10, pin_memory=True)
# dataloader_bd = torch.utils.data.DataLoader(dataset_bd, batch_size=20, num_workers=10, pin_memory=True)
dataloader_sd_test = torch.utils.data.DataLoader(dataset_sd_test, batch_size=20, num_workers=10, pin_memory=True)
dataloader_bd_test = torch.utils.data.DataLoader(dataset_bd_test, batch_size=20, num_workers=10, pin_memory=True)

print("Creating models")

model_sd = Models.RadarP4Transformer(radius=0.5, nsamples=32, spatial_stride=32,
                                    temporal_kernel_size=3, temporal_stride=2,
                                    emb_relu=False,
                                    dim=1024, depth=5, heads=8, dim_head=128,
                                    mlp_dim=2048, num_classes=dataset_sd_test.num_classes)

model_bd = Models.RadarP4Transformer(radius=0.5, nsamples=32, spatial_stride=32,
                                    temporal_kernel_size=3, temporal_stride=2,
                                    emb_relu=False,
                                    dim=1024, depth=5, heads=8, dim_head=128,
                                    mlp_dim=2048, num_classes=dataset_bd_test.num_classes)

# model_combinator = Models.RadarOutputCombinator(2, n_int_layers=0)

if torch.cuda.device_count() > 1:
    model_sd = nn.DataParallel(model_sd)
    model_bd = nn.DataParallel(model_bd)
    # model_combinator = nn.DataParallel(model_combinator)
model_sd = model_sd.to(device)
model_bd = model_bd.to(device)
# model_combinator = model_combinator.to(device)

criterion = nn.CrossEntropyLoss()
# lr = 0.01

# optimizer = torch.optim.SGD(model_combinator.parameters(), lr=lr, momentum=args.momentum)

print("Loading checkpoints")

ckpt_sd = torch.load(args.sd_model_path, map_location='cpu')
model_sd.load_state_dict(ckpt_sd['model'])

ckpt_bd = torch.load(args.bd_model_path, map_location='cpu')
model_bd.load_state_dict(ckpt_bd['model'])

print("Initializing outputs")

# train_sd_out, vid_labels_train, _ = initialize_output(model_sd, dataloader_sd, device, mode=6)
# test_sd_out, vid_labels_test, init_sd_acc = initialize_output(model_sd, dataloader_sd_test, device, mode=6)
# train_bd_out, _, _ = initialize_output(model_bd, dataloader_bd, device, mode=6)
# test_bd_out, _, init_bd_acc = initialize_output(model_bd, dataloader_bd_test, device, mode=6)

# train_out = RadarOutput(vid_labels_train, train_sd_out, train_bd_out)
# test_out = RadarOutput(vid_labels_test, test_sd_out, test_bd_out)

# train_sd_out, ctr_sd_out, vid_labels_train, _ = initialize_output(model_sd, dataloader_sd, device, mode=6)
test_sd_out, cte_sd_out, vid_labels_test, init_sd_acc, init_sd_class_accs = initialize_output(model_sd, dataloader_sd_test, device, mode=6)
# train_bd_out, ctr_bd_out, _, _ = initialize_output(model_bd, dataloader_bd, device, mode=6)
test_bd_out, cte_bd_out, _, init_bd_acc, init_bd_class_accs = initialize_output(model_bd, dataloader_bd_test, device, mode=6)

# train_sd_out = RadarOutput(vid_labels_train, train_sd_out)
# train_bd_out = RadarOutput(vid_labels_train, train_bd_out)
test_sd_out = RadarOutput(vid_labels_test, test_sd_out)
test_bd_out = RadarOutput(vid_labels_test, test_bd_out)

# dataloader_train = torch.utils.data.DataLoader(train_out, batch_size=10, shuffle=True, num_workers=10, pin_memory=True)
# dataloader_test = torch.utils.data.DataLoader(test_out, batch_size=10, num_workers=10, pin_memory=True)

# print("Training combinator")

# epochs = 50
# for epoch in range(epochs):
#     train_one_epoch(model_combinator, criterion, optimizer, dataloader_train, device, epoch)
#     acc, confusion = evaluate(model_combinator, criterion, dataloader_test, device)

# _, _, _, acc, class_acc = fusion_evaluate(train_sd_out, train_bd_out, ctr_sd_out, ctr_bd_out)
# print(acc)
# print(class_acc)

fuse_preds, fuse1_preds, fuse2_preds, acc, class_acc = fusion_evaluate(test_sd_out, test_bd_out, cte_sd_out, cte_bd_out)
print(acc)
print(class_acc)

fuse_preds = { int(k): v.tolist() for k, v in fuse_preds.items() }
fuse1_preds = { int(k): v.tolist() for k, v in fuse1_preds.items() }
fuse2_preds = { int(k): v.tolist() for k, v in fuse2_preds.items() }

if result_file is not None:
    result_file.write('Initial accuracies (0.002, 0.2):\n')
    result_file.write('({}, {})\n'.format(init_sd_acc, init_bd_acc))
    result_file.write('Initial accuracies (0.002 class):\n')
    result_file.write('%s\n'%str(init_sd_class_accs))
    result_file.write('Initial accuracies (0.02 class):\n')
    result_file.write('%s\n'%str(init_bd_class_accs))
    result_file.write('Segment results (fused):\n')
    result_file.write(json.dumps(fuse_preds))
    result_file.write('\n')
    result_file.write('Segment results (0.002):\n')
    result_file.write(json.dumps(fuse1_preds))
    result_file.write('\n')
    result_file.write('Segment results (0.2):\n')
    result_file.write(json.dumps(fuse2_preds))
    result_file.write('\n')
    result_file.write('{}\n'.format(acc))
    result_file.write('%s\n'%str(class_acc))
    # result_file.write('%s\n'%str(confusion.class_recalls().cpu().numpy()))
    result_file.close()
