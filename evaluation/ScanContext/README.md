# ScanContext Baseline

We include evaluation scripts for the [ScanContext](https://github.com/irapkaist/scancontext) baseline on the 5 datasets used in this project. ScanContext is a handcrafted global descriptor which is currently the most popular baseline for outdoor LiDAR-based place recognition. 

As ScanContext does not extract local features, it doesn't support *SpectralGV* integration for re-ranking. However, ScanContext uses it's own re-ranking - see Sec. III.C of the [IROS'2018](https://ieeexplore.ieee.org/document/8593953) paper.

The implementation of ScanContext used in this project is adapted from the partially vectorized implementation by Jacek Komorowski available [here](https://github.com/jac99/Egonn/blob/main/third_party/scan_context/scan_context.py).