# SpectralGV + LoGG3D-Net

- Download the pre-trained model (~103 MB) from Dropbox [here](https://www.dropbox.com/sh/qj5l2dh6gvm81a1/AAA32JqPMnQTuELPodY14xETa?dl=0):
```
cd evaluation/LoGG3D-Net/
wget --output-document logg3d.pth https://dl.dropboxusercontent.com/s/2mghsmkbz8p7ntx/logg3d.pth?dl=0
```
- Evaluate place recognition and metric localization with and without *SpectralGV* re-ranking:
```
python eval_logg3d_sgv.py --dataset_type <dataset_eg_'kitti'> --dataset_root <dataset_root_path>
```

> **Note**: LoGG3D-Net does not support the parallel implementation fo *SpectralGV* as it outputs a varying number of local features/points for each point cloud. 