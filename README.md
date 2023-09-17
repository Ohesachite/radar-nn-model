## Radar NN Model

This model can predict exercises given point cloud sequences derived from mmWave radars.

## Installation

The code is tested with g++ (GCC) 9.4.0, PyTorch v1.12.1, CUDA 11.3.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used for furthest point sampling (FPS) and radius neighbouring search:
```
cd modules
python setup.py install
```

Compilation requires installation of PointNet++ PyTorch version, which can be found [here](https://github.com/erikwijmans/Pointnet2_PyTorch).

## Preprocessing

Preprocessing script can be found in the following path: ``data/radar/process_data.py``

Preprocessing performs DBSCAN and the interpolation and correlation steps of the segmentation

## Classification

To run training for classification:
```
python train-radar.py
```

## Segmentation and Counting

To gather results of the segmentation algorithm:
```
python get-count-accs.py
```