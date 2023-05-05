# SpectralGV + LCDNet

## Usage

  - Clone the [LCDNet](https://github.com/robot-learning-freiburg/LCDNet) codebase into ```evaluation/LCDNet/```.
  ```
  cd evaluation/LCDNet
  git clone https://github.com/robot-learning-freiburg/LCDNet.git
  ```
  - Copy our re-ranking eval script into the EgoNN code base:
  ```
  cp -r SGV_LCDNet/ LCDNet/evaluation/
  cd LCDNet/
  ```
  - Download the pre-trained model (~142 MB) from Dropbox [here](https://www.dropbox.com/sh/qj5l2dh6gvm81a1/AAA32JqPMnQTuELPodY14xETa?dl=0):
  ```
  wget --output-document lcdnet.tar https://dl.dropboxusercontent.com/s/52sis2grvxias7u/lcdnet.tar?dl=0
  mkdir checkpoints && mv lcdnet.tar ./checkpoints/
  ```
  - Evaluate place recognition and metric localization with and without *SpectralGV* re-ranking:
  ```
  python evaluation/SGV_LCDNet/eval_lcdnet_sgv.py --dataset_type <dataset_eg_'kitti'> --dataset_root <dataset_root_path>
  ```