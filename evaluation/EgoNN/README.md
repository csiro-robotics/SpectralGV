# SpectralGV + EgoNN

## Usage

  - Clone the [EgoNN](https://github.com/jac99/Egonn) codebase into this directory.
  ```
  git clone https://github.com/jac99/Egonn.git
  ```
  - Copy our re-ranking eval script into the EgoNN code base:
  ```
  cp -r SGV_EgoNN/ Egonn/eval/
  cd Egonn/eval/SGV_EgoNN/
  ```
  - Evaluate place recognition and metric localization with and without *SpectralGV* re-ranking:
  ```
  python eval_egonn_sgv.py --dataset_type <dataset_eg_'kitti'> --dataset_root <dataset_root_path>
  ```