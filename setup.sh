#!/bin/bash

sudo apt-get update -y
sudo apt-get install -y git wget nano unzip ffmpeg libsm6 libxext6

conda create -n osu3d python=3.10 -y
conda activate osu3d

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y

wget https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2 -O /tmp/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2
conda install /tmp/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2 -y

pip install h5py hydra-core open_clip_torch supervision loguru
pip install wget iopath torchpack pyyaml tqdm
pip install opencv-python natsort imageio onnxruntime
pip install open3d==0.16.0 fast-pytorch-kmeans
conda install -c rapidsai -c nvidia -c conda-forge cuml=24.8 -y
pip install gdown openai
pip install xformers==0.0.22
pip install imageio[ffmpeg]


git clone https://github.com/krrish94/chamferdist.git
cd chamferdist
sed -i 's/c++14/c++17/' setup.py
pip install .
cd ..

git clone https://github.com/gradslam/gradslam.git
cd gradslam
git checkout conceptfusion
git checkout 59ca872e3d265ad09f63c4793d011fad67064452
pip install .
cd ..

git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
cd Grounded-Segment-Anything/segment_anything
pip install -e .
cd ../..

git clone https://github.com/ChaoningZhang/MobileSAM
cd MobileSAM
git checkout c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed
sed -i 's/"onnx",\s*//g; s/"onnxruntime",\s*//g' setup.py
sed -i 's/from \.export import \*//' MobileSAMv2/efficientvit/apps/utils/__init__.py
pip install -e .
cd ..

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
git checkout c121f0432da27facab705978f83c4ada465e46fd
pip install --upgrade timm==0.6.13
sed -i 's/"torch==2\.1\.2",\s*"torchvision==0\.16\.2",\s*//g' pyproject.toml
pip install -e .

git clone https://github.com/facebookresearch/dinov2.git 
cd dinov2
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_linear_head.pth

gdown 1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE -O ~/weights/mobilesamv2/weight.zip
unzip ~/weights/mobilesamv2/weight.zip -d ~/weights/mobilesamv2/
cp ~/MobileSAM/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt ~/weights/mobilesamv2/weight/

echo "export PYTHONPATH=/data/coding/MobileSAM/MobileSAMv2:\$PYTHONPATH" >> ~/.bashrc
echo "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" >> ~/.bashrc
echo "export HF_HUB_OFFLINE=1" >> ~/.bashrc
source ~/.bashrc

# Download data files
mkdir data; cd data; mkdir 3rscan; cd 3rscan; bash preparation.sh; cd ..; cd ..;