# CoMoDaL
## Preparation
### Prerequisites
Tested with
* PyTorch 1.7
* CUDA 10.0
* Python 3.7
* [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)

### Installation
As 3D network we use SparseConvNet. It requires to use CUDA 10.0 (it did not work with 10.1 when we tried).
We advise to create a new conda environment for installation. PyTorch and CUDA can be installed, and SparseConvNet
installed/compiled as follows:
```
$ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
$ pip install --upgrade git+https://github.com/facebookresearch/SparseConvNet.git
```

### Datasets
#### GTA5
Please download the GTA5 dataset from the [GTA5 website](https://download.visinf.tu-darmstadt.de/data/from_games/). The GTA5 dataset directory should have this basic structure:
```
<root_dir>/GTA5/                               % GTA dataset root
<root_dir>/GTA5/images/                        % GTA images
<root_dir>/GTA5/labels/                        % Semantic segmentation labels
...
```

#### SemanticKITTI
Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and
additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip)
from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract
everything into the same folder.

We save all points that project into the front camera image as well
as the segmentation labels to a pickle file.

Please edit the script `data/semantic_kitti/preprocess.py` as follows and then run it.
* `root_dir` should point to the root directory of the SemanticKITTI dataset
* `out_dir` should point to the desired output directory to store the pickle files

## Pre-training 
```
$ python train_source_only.py --cfg=configs/gta_semantic_kitti/train_source_only.yaml
```
Please edit the `OUTPUT_DIR` in the config file, e.g. `configs/gta_semantic_kitti/train_source_only.yaml`,
to choose the output directory.
## Training
### Train CoMoDaL
```
$ python train_CoMoDaL.py --cfg=configs/gta_semantic_kitti/train_CoMoDaL.yaml
```
Please edit the `OUTPUT_DIR` in the config file, e.g. `configs/gta_semantic_kitti/train_CoMoDaL.yaml`,
to choose the output directory.

Please edit the `PRETRAIN_DIR` in the config file, e.g. `configs/gta_semantic_kitti/train_CoMoDaL.yaml`,
to the directory of the pre-trained model.
## Testing
```
$ python test.py --cfg=configs/gta_semantic_kitti/test.yaml
```
Please edit the `OUTPUT_DIR` in the config file, e.g. `configs/gta_semantic_kitti/test.yaml`,
to choose the output directory.

Please edit the `CHECKPOINT2D_DIR` and `CHECKPOINT3D_DIR` in the config file, e.g. `configs/gta_semantic_kitti/test.yaml`,
to the directory of the 2D model and 3D model trained with CoMoDaL.


