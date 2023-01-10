# mSME
A rapid surface extraction algorithm in Python.

mSME is able to extract a 
comprehensive and continuous surface from the 3D volume, with improvements in performance speed, and usability.
Implemention based on FastSME algorithm: [Basu et al.](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w44/Basu_FastSME_Faster_and_CVPR_2018_paper.pdf)

Full methods described in: G. Gao, D. Miyasato, L.A. Barner, R. Serafin, K.W. Bishop, W. Xie, A.K. Glaser, E. L. Rosenthal, L.D. True, and J.T.C. Liu, “Comprehensive surface histology of fresh resection margins with rapid open-top light-sheet (OTLS) microscopy,” IEEE Trans. Biomed. Eng. [in press]


## Installation

In order to use mSME's GPU acceleration, install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive).

Clone this repository:

```bash
git clone https://github.com/alecgao066/mSME.git
cd mSME
```

Install all dependencies at a new conda environment (surface_extraction):

```bash
conda env create -f environment.yml
```


## Usage
See example.ipynb to get started.

Example image stack provided via shared google drive link.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.