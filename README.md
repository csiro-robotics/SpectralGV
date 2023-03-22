# SpectralGV
This repository is an open-source implementation of the RA-L paper: [Spectral Geometric Verification: Re-Ranking Point Cloud Retrieval for Metric Localization](https://ieeexplore.ieee.org/document/10065560). 

Paper Pre-print: https://arxiv.org/abs/2210.04432

## Method overview.
*SpectralGV* is an efficient spectral method for geometric verification based re-ranking of point clouds. *SpectralGV* allows integration with any metric-localization architecture that extracts both local and global features for a given point cloud, without modifying the architectures and without further training of learning-based methods. 

In this repository, we provide code and pretrained models for integrating *SpectralGV* with [EgoNN](https://ieeexplore.ieee.org/document/9645340) (RA-L'2022) and [LCDNet](https://ieeexplore.ieee.org/document/9723505) (T-Ro'2022), and [LoGG3D-Net](https://ieeexplore.ieee.org/document/9811753) (ICRA'2022).

![](./utils/docs/SGV_pipeline.png)


### UPDATES
- [ ] Add integration with EgoNN
- [ ] Add integration with LCDNet
- [ ] Add integration with LoGG3D-Net

## Usage

### Set up environment
This project has been tested on Ubuntu 22.04. Set up the requirments as follows:
- Create [conda](https://docs.conda.io/en/latest/) environment with pytorch and open3d:
```bash
conda create --name sgv_env python=3.9.4
conda activate sgv_env
```
- Install PyTorch with suitable cudatoolkit version. See [here](https://pytorch.org/):
```bash
pip3 install torch torchvision torchaudio
# Make sure the pytorch cuda version matches your output of 'nvcc --version'
```




<details>
  <summary><b>Add dependencies for LoGG3D-Net:</b></summary><br/>
  
    - Install [Open3d](https://github.com/isl-org/Open3D), [Torchpack](https://github.com/zhijian-liu/torchpack):
    ```bash
    pip install -r requirements.txt
    ```
    - Install torchsparse-1.4.0
    ```bash
    sudo apt-get install libsparsehash-dev
    pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
    ```
    - Install [mpi4py](https://mpi4py.readthedocs.io/en/stable/tutorial.html):
    ```bash
    conda install mpi4py
    conda install openmpi
    ```
  
</details>


<details>
  <summary><b>Add dependencies for EgoNN:</b></summary><br/>

</details>

<details>
  <summary><b>Add dependencies for LCDNet:</b></summary><br/>

</details>

### Evaluation


## Citation

If you find this work useful in your research, please cite:

```
@article{vidanapathirana2023sgv,
  author={Vidanapathirana, Kavisha and Moghadam, Peyman and Sridharan, Sridha and Fookes, Clinton},
  journal={IEEE Robotics and Automation Letters}, 
  title={Spectral Geometric Verification: Re-Ranking Point Cloud Retrieval for Metric Localization}, 
  year={2023},
  publisher={IEEE},
  volume={8},
  number={5},
  pages={2494-2501},
  doi={10.1109/LRA.2023.3255560}}
```


## Acknowledgement
Functions from 3rd party have been acknowledged at the respective function definitions or readme files. This project was mainly inspired by the following: [SpectralMatching](https://ieeexplore.ieee.org/document/1544893) and [PointDSC](https://github.com/XuyangBai/PointDSC).

## Contact
For questions/feedback, 
 ```
 kavisha.vidanapathirana@data61.csiro.au
 ```
