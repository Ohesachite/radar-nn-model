# RF-HAC

View-agnostic Human Exercise Cataloging with Single MmWave Radar

This repo contains the implementation of the paper: [View-agnostic Human Exercise Cataloging with Single MmWave Radar](https://dl.acm.org/doi/10.1145/3678512) published in Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, Volume 8, Issue 3.

**Authors**

Alan Liu, YuTai Lin, Karthik Sundaresan

## Installation

The code is tested with g++ (GCC) 9.4.0, PyTorch v1.12.1, CUDA 11.3.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413):
```
cd modules
python setup.py install
```

Compilation requires installation of PointNet++ PyTorch version, which can be found [here](https://github.com/erikwijmans/Pointnet2_PyTorch).

## Data

Place raw data in folder ``data/radar`` before running preprocessing. There should only be one level of subdirectories between ``data/radar`` and the csv files, e.g. a given raw data file should have a path like this: ``data/radar/set8_0/label_aud_01a.csv``

Checkpoints can be found in ``ckpts`` folder.

The columns of each of the csv files should have the following data: 
``frame_time, frame_id, point_id, x, y, z, doppler, intensity``

## Preprocessing

Preprocessing script can be found in the following path: ``data/radar/process_data_review.py``

## Classification

To run training for classification:
```
python train-radar.py
```

Use ``--train-path=`` and ``--test-path=`` to specify the folder for training and testing respectively (after preprocessing).

Use ``--resume=`` and specify the path to the checkpoint to load a checkpoint. Loading a checkpoint will result in training script automatically skipping to final inference.

Use ``--output-dir=`` when training and specify a folder to save checkpoints during training.

## Segmentation and Counting

To gather results of the segmentation algorithm:
```
python get-count-accs.py
```
Use ``--sets`` flag to specify a list of folders containing raw data
