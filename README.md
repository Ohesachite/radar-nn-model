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

Raw data collected by us can be found [here](https://zenodo.org/records/10602471). The datasets consists of Environment 2 under folder  ``setenv-457_test`` and Unseen Environments ``env935, env935_60, env85, env_131_0, env_131_nonfront``.

Place raw data in folder ``data/radar`` before running preprocessing. There should only be one level of subdirectories between ``data/radar`` and the csv files, e.g. a given raw data file should have a path like this: ``data/radar/set8_0/label_aud_01a.csv``

Checkpoints can be found in ``ckpts`` folder.

The columns of each of the csv files should have the following data: 
``frame_time, frame_id, point_id, x, y, z, doppler, intensity``

## Preprocessing

Preprocessing script can be found in the following path: ``data/radar/process_data.py``

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

## Cite

If you used this code in any of your projects, you can cite our paper with the following BibTex:

```
@article{10.1145/3678512,
author = {Liu, Alan and Lin, Yu-Tai and Sundaresan, Karthikeyan},
title = {View-agnostic Human Exercise Cataloging with Single MmWave Radar},
year = {2024},
issue_date = {August 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {8},
number = {3},
url = {https://doi.org/10.1145/3678512},
doi = {10.1145/3678512},
abstract = {Advances in mmWave-based sensing have enabled a privacy-friendly approach to pose and gesture recognition. Yet, providing robustness with the sparsity of reflected signals has been a long-standing challenge towards its practical deployment, constraining subjects to often face the radar. We present RF-HAC- a first-of-its-kind system that brings robust, automated and real-time human activity cataloging to practice by not only classifying exercises performed by subjects in their natural environments and poses, but also tracking the corresponding number of exercise repetitions. RF-HAC's unique approach (i) brings the diversity of multiple radars to scalably train a novel, self-supervised, pose-agnostic transformer-based exercise classifier directly on 3D RF point clouds with minimal manual effort and be deployed on a single radar; and (ii) leverages the underlying doppler behavior of exercises to design a robust self-similarity based segmentation algorithm for counting the repetitions in unstructured RF point clouds. Evaluations on a comprehensive set of challenging exercises in both seen and unseen environments/subjects highlight RF-HAC's robustness with high accuracy (over 90\%) and readiness for real-time, practical deployments over prior art.},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = sep,
articleno = {117},
numpages = {23},
keywords = {human activity recognition, mmWave Sensing, self-supervised learning, view-agnostic sensing}
}
```
