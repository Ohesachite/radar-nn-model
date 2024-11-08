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

MAX_POSITIVE_KEYPOINT_MPJPE = None

def train_one_epoch(model, criterion, optimizer, contrastive_optimizer, lr_scheduler, data_loader, device, epoch, print_freq, contrastive_alpha=0.3, contrastive_weight=0.1):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Train: [{}]'.format(epoch)
    # add negative clip and feature
    for clip, features, target, _, _, _, positive_clip, positive_features in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        clip, features, target = clip.to(device), features.to(device), target.to(device)
        positive_clip, positive_features = positive_clip.to(device), positive_features.to(device)
        output, xyzts, features = model(clip, features)
        _, _, positive_features = model(positive_clip, positive_features)
        loss = criterion(output, target)
        positive_indicator_matrix = losses.compute_positive_indicator_matrix(target, device)
        anchor_representation_loss = (1.0 - contrastive_alpha) * losses.compute_representation_loss(xyzts, (xyzts, features), losses.TYPE_FUSION_OP_MOE, positive_indicator_matrix)
        view_loss = contrastive_alpha * losses.compute_fenchel_dual_loss(features, positive_features, losses.TYPE_MEASURE_JSD, positive_indicator_matrix=positive_indicator_matrix)

        contrastive_loss = contrastive_weight * (anchor_representation_loss + view_loss)
        # loss = loss + contrastive_loss

        optimizer.zero_grad()
        contrastive_optimizer.zero_grad()

        loss.backward(retain_graph=True)
        contrastive_loss.backward()

        optimizer.step()
        contrastive_optimizer.step()

        acc1, acc3 = utils.accuracy(output, target, topk=(1, 3))
        batch_size = clip.shape[0]
        metric_logger.update(trans_loss=loss.item(), con_loss=contrastive_loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()
        sys.stdout.flush()

def validate(model, criterion, data_loader, device, epoch, print_freq, contrastive_alpha=0.3, contrastive_weight=0.1):
    if data_loader is None:
        return

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation: [{}]'.format(epoch)
    with torch.no_grad():
        # add negative clip and features
        for clip, features, target, _, _, _, positive_clip, positive_features in metric_logger.log_every(data_loader, print_freq, header):
            clip, features, target = clip.to(device), features.to(device), target.to(device)
            positive_clip, positive_features = positive_clip.to(device), positive_features.to(device)
            output, xyzts, features = model(clip, features)
            _, _, positive_features = model(positive_clip, positive_features)
            loss = criterion(output, target)
            positive_indicator_matrix = losses.compute_positive_indicator_matrix(target, device)
            anchor_representation_loss = (1.0 - contrastive_alpha) * losses.compute_representation_loss(xyzts, (xyzts, features), losses.TYPE_FUSION_OP_MOE, positive_indicator_matrix)
            view_loss = contrastive_alpha * losses.compute_fenchel_dual_loss(features, positive_features, losses.TYPE_MEASURE_JSD, positive_indicator_matrix=positive_indicator_matrix)

            contrastive_loss = contrastive_weight * (anchor_representation_loss + view_loss)

            acc1, acc3 = utils.accuracy(output, target, topk=(1, 3))
            batch_size = clip.shape[0]
            metric_logger.update(trans_loss=loss.item(), con_loss=contrastive_loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    print(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@3 {top3.global_avg:.3f}'.format(top1=metric_logger.acc1, top3=metric_logger.acc3))


def evaluate(model, criterion, data_loader, device, mode=0):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    confusion = utils.ConfusionMatrix(10, device)
    header = 'Test:'
    video_prob = {}
    video_label = {}
    with torch.no_grad():
        for clip, features, target, video_idx, vid_samp_n, vid_n_samp, _, _ in metric_logger.log_every(data_loader, 100, header):
            clip = clip.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output, _, _ = model(clip, features)
            loss = criterion(output, target)

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
                    video_prob[idx] = np.zeros((vid_n_samp[i], data_loader.dataset.num_classes))
                    video_prob[idx][vid_samp_n[i], :] += prob[i]
                    video_label[idx] = target[i]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@3 {top3.global_avg:.3f}'.format(top1=metric_logger.acc1, top3=metric_logger.acc3))

    confusion_matrix = confusion.get_confusion_matrix().cpu().numpy()
    precision = confusion.class_precisions().cpu().numpy()
    recall = confusion.class_recalls().cpu().numpy()

    print('Confusion Matrix:')
    with np.printoptions(precision=3):
        print(confusion_matrix)
        print(precision)
        print(recall)

    # video level prediction
    video_pred = {k: np.argmax(np.sum(v, axis=0)) for k, v in video_prob.items()}
    pred_correct = [video_pred[k]==video_label[k] for k in video_pred]
    total_acc = np.mean(pred_correct)

    class_count = [0] * data_loader.dataset.num_classes
    class_correct = [0] * data_loader.dataset.num_classes

    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += (v==label)
    class_acc = [c/float(s) for c, s in zip(class_correct, class_count)]

    print(' * Video Acc@1 %f'%total_acc)
    print(' * Class Acc@1 %s'%str(class_acc))

    # Subsample level prediction
    samp_confusion = utils.ConfusionMatrix(10, 'cpu')
    if mode == 0:
        samp3_prob = {}
        samp5_prob = {}
        clip_len = data_loader.dataset.frames_per_clip
        for k, v in video_prob.items():
            prob3_size = v.shape[0] - 2
            if prob3_size > 0:
                samp3_prob[k] = v[:prob3_size,:] + v[1:1+prob3_size,:] + v[2:2+prob3_size,:]
            else:
                samp3_prob[k] = v

            prob5_size = v.shape[0] - 4
            if prob5_size > 0:
                samp5_prob[k] = v[:prob5_size,:] + v[1:1+prob5_size,:] + v[2:2+prob5_size,:] + v[3:3+prob5_size,:] + v[4:4+prob5_size,:]
            else:
                samp5_prob[k] = v

        samp3_pred = {k: np.argmax(v, axis=1) for k, v in samp3_prob.items()}
        pred3_correct = [samp3_pred[k][i]==video_label[k] for k in samp3_pred for i in range(len(samp3_pred[k]))]
        samp3_acc = np.mean(pred3_correct)

        for k, v in samp3_pred.items():
            for v2 in v:
                samp_confusion.add_individual_pred(torch.tensor(video_label[k]), torch.tensor(v2))

        print(' * 3 Samples Acc@1 %f'%samp3_acc)
        print(' * 3 Samples Acc By Class@1 %s'%str(samp_confusion.class_precisions().numpy()))

        samp5_pred = {k: np.argmax(v, axis=1) for k, v in samp5_prob.items()}
        pred5_correct = [samp5_pred[k][i]==video_label[k] for k in samp5_pred for i in range(len(samp5_pred[k]))]
        samp5_acc = np.mean(pred5_correct)

        class_count = [0] * data_loader.dataset.num_classes
        class_correct = [0] * data_loader.dataset.num_classes

        for k, v in samp5_pred.items():
            label = video_label[k]
            for v2 in v:
                class_count[label] += 1
                class_correct[label] += (v2==label)
        class_acc = [c/float(s) for c, s in zip(class_correct, class_count)]

        print(' * 5 Samples Acc@1 %f'%samp5_acc)
        print(' * 5 Samples Acc By Class@1 %s'%str(class_acc))

        return total_acc, samp3_acc, samp_confusion, metric_logger.meters['acc1'].global_avg, confusion
    
    elif mode == 3 or mode == 5 or mode == 6:
        seg_pred = {k: np.argmax(v, axis=1) for k, v in video_prob.items()}
        predseg_correct = [seg_pred[k][i] == video_label[k] for k in seg_pred for i in range(len(seg_pred[k]))]
        seg_acc = np.mean(predseg_correct)

        for k, v in seg_pred.items():
            label = video_label[k]
            for v2 in v:
                samp_confusion.add_individual_pred(torch.tensor(label), torch.tensor(v2))

        print(' * Segmented Clip Acc@1 %f'%seg_acc)
        print(' * Segment Clip Acc by Class@1 %s'%str(samp_confusion.class_precisions().numpy()))

        return total_acc, seg_acc, samp_confusion, metric_logger.meters['acc1'].global_avg, confusion


def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    if args.result_file:
        result_file = open(args.result_file, "w")
    else:
        result_file = None

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    device = torch.device('cuda')

    # Data loading code
    print("Loading data")

    st = time.time()

    dataset = Radar(
            root=args.train_path,
            frames_per_clip=args.clip_len,
            frame_interval=args.frame_interval,
            num_points=args.num_points,
            mask_split=[0.8, 0.2, 0.0],
            mode=args.train_mode,
            decay=args.decay
    )

    dataset_test = Radar(
            root=args.test_path,
            frames_per_clip=args.clip_len,
            frame_interval=args.frame_interval,
            num_points=args.num_points,
            mask_split=[0.0, 0.0, 1.0],
            mode=args.test_mode
    )

    print("Creating data loaders")

    subset_train = torch.utils.data.Subset(dataset, dataset.train_indices)
    data_loader_train = torch.utils.data.DataLoader(subset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    has_val = len(dataset.val_indices) != 0
    if has_val:
        subset_val = torch.utils.data.Subset(dataset, dataset.val_indices)
        data_loader_val = torch.utils.data.DataLoader(subset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    else:
        data_loader_val = None

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = Models.RadarP4Transformer(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                                    temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                                    emb_relu=args.emb_relu,
                                    dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head,
                                    mlp_dim=args.mlp_dim, num_classes=dataset.num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if torch.cuda.device_count() > 1:
        contrastive_optimizer = torch.optim.Adagrad([
            {'params': model.module.tube_embedding.parameters()}, 
            {'params': model.module.pos_embedding.parameters()}
            ], lr=lr, weight_decay=args.weight_decay)
    else:
        contrastive_optimizer = torch.optim.Adagrad([
            {'params': model.tube_embedding.parameters()}, 
            {'params': model.pos_embedding.parameters()}
            ], lr=lr, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
    lr_milestones = [len(data_loader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    if args.resume:
        print("Loading checkpoint")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1


    print("Start training")
    start_time = time.time()
    acc = 0
    epoch_acc = 0
    samp_acc = 0 
    clip_acc = 0 
    if args.epochs > args.start_epoch:
        for epoch in range(args.start_epoch, args.epochs):

            train_one_epoch(model, criterion, optimizer, contrastive_optimizer, lr_scheduler, data_loader_train, device, epoch, args.print_freq, contrastive_alpha=args.contrastive_alpha, contrastive_weight=args.contrastive_weight)

            validate(model, criterion, data_loader_val, device, epoch, args.print_freq, contrastive_alpha=args.contrastive_alpha, contrastive_weight=args.contrastive_weight)

            epoch_acc, samp_acc, samp_confusion, clip_acc, clip_confusion = evaluate(model, criterion, data_loader_test, device=device, mode=args.test_mode)

            acc = max(acc, epoch_acc)

            if args.output_dir:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'ckpt_{}.pth'.format(epoch)))
    else:
        acc, samp_acc, samp_confusion, clip_acc, clip_confusion = evaluate(model, criterion, data_loader_test, device=device, mode=args.test_mode)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Accuracy {}'.format(acc))

    if result_file is not None:
        result_file.write('Final accuracies (sample acc, sample pre by class, sample rec by class, clip acc, clip pre by class, clip rec by class):\n')
        result_file.write('{}\n'.format(samp_acc))
        result_file.write('%s\n'%str(samp_confusion.class_precisions().numpy()))
        result_file.write('%s\n'%str(samp_confusion.class_recalls().numpy()))
        result_file.write('{}\n'.format(clip_acc))
        result_file.write('%s\n'%str(clip_confusion.class_precisions().cpu().numpy()))
        result_file.write('%s\n'%str(clip_confusion.class_recalls().cpu().numpy()))
        result_file.close()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')

    parser.add_argument('--train-path', default='', type=str, help='training dataset')
    parser.add_argument('--test-path', default='', type=str, help='testing dataset')
    parser.add_argument('--train-mode', default=0, type=int, help='training mode number')
    parser.add_argument('--test-mode', default=0, type=int, help='testing mode number')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    # input
    parser.add_argument('--clip-len', default=24, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--frame-interval', default=1, type=int, metavar='N', help='interval of sampled frames')
    parser.add_argument('--num-points', default=1024, type=int, metavar='N', help='number of points per frame')
    parser.add_argument('--decay', default=0.002, type=float, help='segmentation decay factor')
    # P4D
    parser.add_argument('--radius', default=0.5, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=2, type=int, help='temporal stride')
    # embedding
    parser.add_argument('--emb-relu', default=False, action='store_true')
    # transformer
    parser.add_argument('--dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--depth', default=5, type=int, help='transformer depth')
    parser.add_argument('--heads', default=8, type=int, help='transformer head')
    parser.add_argument('--dim-head', default=128, type=int, help='transformer dim for each head')
    parser.add_argument('--mlp-dim', default=2048, type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch-size', default=20, type=int)
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--contrastive-alpha', default=0.1, type=float, help='view loss weight in contrastive loss')
    parser.add_argument('--contrastive-weight', default=0.1, type=float, help='multiplier for contrastive loss')
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', type=str, help='path where to save')
    parser.add_argument('--result-file', default='', type=str, help='path to save accuracies')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
