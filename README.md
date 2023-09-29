# Crossloc3D release
ICCV 2023: CrossLoc3D: Aerial-Ground Cross-Source 3D Place Recognition 

[CS-Campus3D Dataset](https://drive.google.com/file/d/1rFwfK3LxjMQnzlG_v_73dk63KyphnNjy/view?usp=sharing)


# Setting up Environment

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




# Dataset


## Oxford RobotCar dataset

Follow instruction of this [repo](https://github.com/mikacuy/pointnetvlad) or download benchmark_datasets.zip from [here](https://drive.google.com/drive/folders/1Wn1Lvvk0oAkwOUwR0R6apbrekdXAUg7D) and put /benchmark_datasets folder in /data folder.


Run:
python ./datasets/preprocess/generate_training_tuples_baseline.py
python ./datasets/preprocess/generate_test_sets.py


## CS-Campus3D (Ours)

Download data [here](https://drive.google.com/file/d/1rFwfK3LxjMQnzlG_v_73dk63KyphnNjy/view?usp=sharing) and put /benchmark_datasets folder in /data folder.


# Training

python main.py ./configs/<config_file>.py
