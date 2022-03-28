ARG PYTORCH="1.6.0"
# ARG PYTORCH="1.8.0"
ARG CUDA="10.1"
# ARG CUDA="11.1"
# ARG CUDNN="8"
ARG CUDNN="7"
ARG MMCV="1.3.0"
# ARG MMCV="1.4.4"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && apt-get install -y libgl1-mesa-glx
RUN pip install natsort pycm 

RUN conda clean --all

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]

ENV FORCE_CUDA="1"

# COPY ./ /multimodal

# # Install MMSegmentation
# RUN git clone https://github.com/open-mmlab/mmsegmentation.git /multimodal/mmsegmentation
# WORKDIR /multimodal/mmsegmentation

# RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -e .

# WORKDIR /multimodal

# RUN cp petraw.py /multimodal/mmsegmentation/mmseg/datasets/
# RUN cp __init__.py /multimodal/mmsegmentation/mmseg/datasets/
# RUN cp deeplabv3_plus.py /multimodal/mmsegmentation/tools/
# RUN cp train_deeplabv3.sh /multimodal/mmsegmentation/tools/

# WORKDIR /multimodal/mmsegmentation