# CrossLoc3D: Aerial-Ground Cross-Source 3D Place Recognition (Accepted by ICCV 2023)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/rayguan97/crossloc3d/.github%2Fworkflows%2Fpython-package-conda.yml)
[![GitHub Repo stars](https://img.shields.io/github/stars/rayguan97/crossloc3d)](https://github.com/rayguan97/crossloc3d/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/rayguan97/crossloc3d)](https://github.com/rayguan97/crossloc3d/network)
[![GitHub issues](https://img.shields.io/github/issues/rayguan97/crossloc3d)](https://github.com/rayguan97/crossloc3d/issues)
[![GitHub](https://img.shields.io/github/license/rayguan97/crossloc3d)](https://github.com/rayguan97/crossloc3d/blob/master/LICENSE)


[CrossLoc3D: Aerial-Ground Cross-Source 3D Place Recognition](https://arxiv.org/abs/2303.17778)
<br> Tianrui Guan, Aswath Muthuselvam, Montana Hoover, Xijun Wang, Jing Liang, Adarsh Jagan Sathyamoorthy, Damon Conover, Dinesh Manocha

# Motivation
<img src="./resources/diff.png" width="1080">
<br>
Representation gap between aerial and ground sources: We use the bounding box with the same color to focus on the same region and highlight the differences between aerial (**left**) and ground (**right**) LiDAR scans. Scopes (<span style="color:cyan">cyan</span>): The aerial scans cover a large region, while ground scans cover only a local area. Coverages (<span style="color:lawngreen">green</span>): The aerial scans cover the top of the buildings, while ground scans cover more details on the ground. Densities (<span style="color:blue">blue</span>): The distribution and density of the points are different because of various scan patterns, effective ranges, and fidelity of LiDARs. Noise Patterns (<span style="color:red">red</span>): The aerial scans have larger noises, as we can see from a bird-eye view and top-down view of a corner of the building.
<br>
<br>

# Network Architecture
<img src="./resources/network.png" width="1080">
<br>
<br>

If you find this project useful in your research, please cite our work:

```latex
@InProceedings{Guan_2023_ICCV,
    author    = {Guan, Tianrui and Muthuselvam, Aswath and Hoover, Montana and Wang, Xijun and Liang, Jing and Sathyamoorthy, Adarsh Jagan and Conover, Damon and Manocha, Dinesh},
    title     = {CrossLoc3D: Aerial-Ground Cross-Source 3D Place Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
}

```


# Getting Started
## Setting up Environment

```
conda create -n crossloc python=3.7 pandas tensorboard numpy -c conda-forge
conda activate crossloc
conda install pytorch=1.9.1 torchvision cudatoolkit=11.1 -c pytorch -c nvidia


conda install openblas-devel -c anaconda
sudo apt-get install openexr and libopenexr-dev
conda install -c conda-forge openexr

pip install laspy pytest addict pytorch-metric-learning==0.9.97 yapf==0.40.1 bitarray==1.6.0 h5py transforms3d open3d
pip install tqdm setuptools==59.5.0 einops
pip install bagpy utm pptk
conda install openexr-python
pip install pyexr pyntcloud


cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

## Dataset


### Oxford RobotCar dataset

Follow instruction of this [repo](https://github.com/mikacuy/pointnetvlad) or download benchmark_datasets.zip from [here](https://drive.google.com/drive/folders/1Wn1Lvvk0oAkwOUwR0R6apbrekdXAUg7D) and put /benchmark_datasets folder in /data folder.


```
python ./datasets/preprocess/generate_training_tuples_baseline.py
python ./datasets/preprocess/generate_test_sets.py
```


### CS-Campus3D (Ours)

The dataset can be accessed [here](https://drive.google.com/file/d/1rFwfK3LxjMQnzlG_v_73dk63KyphnNjy/view?usp=sharing).

Download data and put /benchmark_datasets folder in /data folder.


## Training

```
python main.py ./configs/<config_file>.py
```


## Evaluation

```
python main.py ./configs/<config_file>.py --mode val --resume_from <ckpt_location>.pth
```

### Checkpoints

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Dataset</th>
<th valign="bottom">config</th>
<th valign="bottom">ckpt</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_50ep -->
 <tr>
    <td align="left">Crossloc3D</td>
    <td align="center">Oxford</td>
    <td align="left"><a href="https://github.com/rayguan97/crossloc3d/blob/main/configs/oxford_ours.py">config</a></td>
    <td align="left"><a href="https://drive.google.com/file/d/1DC-6s0byNX3xL9FYpGRkNnlZSZdbjNag/view?usp=sharing">ckpt</a></td>
 </tr>
 <tr>
    <td align="left">Crossloc3D</td>
    <td align="center">CS-Campus3D</td>
    <td align="left"><a href="https://github.com/rayguan97/crossloc3d/blob/main/configs/campus_ours.py">config</a></td>
    <td align="left"><a href="https://drive.google.com/file/d/1gbT6QDLgvbxkNb3JsGBDtPwHE3K_idCv/view?usp=sharing">ckpt</a></td>
 </tr>

</tbody></table>
