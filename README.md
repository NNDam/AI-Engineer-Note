# AI-Engineer-Note

Tất cả những thứ sưu tầm được liên quan đến AI Engineer và Deploy Services 


- [Deeplearning](Deeplearning)
    + [1. ComputerVision](Deeplearning/ComputerVision)
        + [1.1 Common Architectures](Deeplearning/ComputerVision)
            + [1.1.1 ResBlock](Deeplearning/ComputerVision/docs/resblock.md)
            + [1.1.2 Gated Convolution](Deeplearning/ComputerVision/docs/gated_convolution.md)
            + [1.1.3 Multi-head Attention](Deeplearning/ComputerVision/docs/multihead_attn.md)
    + [2. NLP](Deeplearning/NLP)
- [Frameworks](Framework)
    + [1. TensorRT](Framework/TensorRT)
        + [1.1 Convert ONNX model to TensorRT](Framework/TensorRT/docs/tutorial.md)
        + [1.2 Wrapped TensorRT-CPP Models](https://github.com/NNDam/TensorRT-CPP)
            + [1.2.1 Arcface](https://github.com/NNDam/TensorRT-CPP/tree/main/Arcface)
            + [1.2.2 SCRFD](https://github.com/NNDam/TensorRT-CPP/tree/main/SCRFD)
            + [1.2.3 YOLOv7](https://github.com/NNDam/TensorRT-CPP/tree/main/YOLOv7)
    + [2. Pytorch](Framework/Pytorch)
        + [2.1 Build Pytorch from source (Optimize speed for AMD CPU & NVIDIA GPU)](Framework/Pytorch/docs/build_from_source.md)
- [Deploy](Deploy)
    + [1. NVIDIA](Deploy/NVIDIA)
        + [1.1 Multi-instance GPU (MIG)](Deploy/NVIDIA/docs/multi_instance_gpu.md)
        + [1.2 FFMPEG with Nvidia hardware-acceleration](Deploy/NVIDIA/docs/nvidia_video_sdk.md)
    + [2. Deepstream](Deploy/Deepstream)
        + [2.1 Yolov4](Deploy/Deepstream/sample-yolov4)
        + [2.2 Traffic Analyst](Deploy/Deepstream/sample-ALPR)
        + [2.3 SCRFD Face Detection (custom parser & NMS plugin with landmark)](Deploy/Deepstream/sample-scrfd)
    + [3. Triton Inference Server](Deploy/Triton-inference-server)
        - [3.1 Cài đặt triton-server và triton-client](Deploy/Triton-inference-server/docs/install.md)
            + [3.1.1 Các chế độ quản lý model (load/unload/reload)](Deploy/Triton-inference-server/docs/model_management.md)
        - [3.2 Sơ lược về các backend trong Triton](Deploy/Triton-inference-server/docs/backend.md)
        - [3.3 Cấu hình cơ bản khi deploy mô hình](Deploy/Triton-inference-server/docs/model_configuration.md)
        - [3.4 Deploy mô hình](#)
            - [3.4.1 ONNX-runtime](Deploy/Triton-inference-server/docs/triton_onnx.md)
            - [3.4.2 TensorRT](Deploy/Triton-inference-server/docs/triton_tensorrt.md)
            - [3.4.3 Pytorch & TorchScript](Deploy/Triton-inference-server/docs/triton_pytorch.md)
            - [3.4.4 Kaldi <i>(Advanced)</i>](Deploy/Triton-inference-server/docs/triton_kaldi.md)
        - [3.5 Model Batching](Deploy/Triton-inference-server/docs/model_batching.md)
        - [3.6 Ensemble Model và pre/post processing](Deploy/Triton-inference-server/docs/model_ensemble.md)
        - [3.7 Sử dụng Performance Analyzer Tool](Deploy/Triton-inference-server/docs/perf_analyzer.md)
        - [3.8 Optimizations](#)
            + [3.8.1 Tối ưu Pytorch backend](Deploy/Triton-inference-server/docs/optimization_pytorch.md)
    + [4. TAO Toolkit (Transfer-Learning-Toolkit)](Deploy/Transfer-Learning-Toolkit)

- [Linux & CUDA & APT-Packages](Linux)
    + <details><summary><b>Build OpenCV from source</b></summary>

        - [Build OpenCV from source](docs/build_opencv.md)
        
        </details>

    + <details><summary><b>Install Math Kernel Library (MKL/BLAS/LAPACK/OPENBLAS)</b></summary>
        You are recommended to install all Math Kernel Library and then compile framework (e.g pytorch, mxnet) from source using custom config for optimization.</br>
        Install all LAPACK+BLAS:
        
        ```
        sudo apt install libjpeg-dev libpng-dev libblas-dev libopenblas-dev libatlas-base-dev liblapack-dev liblapacke-dev gfortran 
        ```
        
        Install MKL:
        
        ```
        # Get the key
        wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
        # now install that key
        apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
        # now remove the public key file exit the root shell
        rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
        # Add to apt
        sudo wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
        sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
        # Install
        sudo apt-get update
        sudo apt-get install intel-mkl-2020.4-912
        ```
        
        </details>

    + <details><summary><b>Fresh install NVIDIA driver (PC/Laptop/Workstation)</b></summary>

        ```
        # Remove old packages
        sudo apt-get remove --purge '^nvidia-.*'
        sudo apt-get install ubuntu-desktop
        sudo apt-get --purge remove "*cublas*" "cuda*"
        sudo apt-get --purge remove "*nvidia*"
        sudo add-apt-repository --remove ppa:graphics-drivers/ppa
        sudo rm /etc/X11/xorg.conf
        sudo apt autoremove
        sudo reboot
        
        # After restart
        sudo ubuntu-drivers devices
        sudo ubuntu-drivers autoinstall
        sudo reboot
        ```
        
        </details>

    + <details><summary><b>NVIDIA-SMI has failed because it couldn’t communicate with the NVIDIA driver</b></summary>
  
        First, make sure that you have "Fresh install NVIDIA driver". If not work, try this bellow
          
        - Make sure the package nvidia-prime is installed:
        
        ```
        sudo apt install nvidia-prime
        ```
        
        Afterwards, run
        ```
        sudo prime-select nvidia
        ```
        
        - Make sure that NVIDIA is not in blacklist
          
        ```
        grep nvidia /etc/modprobe.d/* /lib/modprobe.d/*
        ```
        
        to find a file containing ```blacklist nvidia``` and remove it, then run
        
        ```
        sudo update-initramfs -u
        ```
        
        - If get error ```This PCI I/O region assigned to your NVIDIA device is invalid```:
        
        ```
        sudo nano /etc/default/grub
        ```
        
        edit ```GRUB_CMDLINE_LINUX_DEFAULT="quiet splash pci=realloc=off"```
        
        ```
        sudo update-grub
        sudo reboot
        ```
        
        </details>

    + <details><summary><b>Check current CUDA version</b></summary>

        ```
        nvcc --version
        ```
        
        </details>

    + <details><summary><b>Check current supported CUDA versions</b></summary>

        ```
        ls /usr/local/
        ```
        
        </details>

    + <details><summary><b>Select GPU devices</b></summary>

        ```
        CUDA_VISIBLE_DEVICES=<index-of-devices> <command>
        CUDA_VISIBLE_DEVICES=0 python abc.py
        CUDA_VISIBLE_DEVICES=0 ./sample.sh
        CUDA_VISIBLE_DEVICES=0,1,2,3 python abc.py
        CUDA_VISIBLE_DEVICES=0,1,2,3 ./sample.sh
        ```
        
        </details>

    + <details><summary><b>Switch CUDA version</b></summary>

        ```
        CUDA_VER=11.3
        export PATH="/usr/local/cuda-$CUDA_VER/bin:$PATH"
        export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VER/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
        ```
        
        </details>

    + <details><summary><b>Check NVENV/NVDEC status</b></summary>
        
        ```
        nvidia-smi dmon
        ```
        see the tab **%enc** and **%dec**
        </details>

    + <details><summary><b>Error with distributed training NCCL (got freezed)</b></summary>
        
        ```
        export NCCL_P2P_DISABLE="1"
        ```
        
        </details>

    + <details><summary><b>Broken pipe (Distributed training with NCCL)</b></summary>
        Run training with args
        
        ```
        NCCL_DEBUG=INFO TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=INFO torchrun ...
        ```
        
        to gather **socket name** (e.g ```eno1```)
      
        ```
        NCCL INFO NET/IB : No device found.
        rnd3:77634:79720 [0] NCCL INFO NET/Socket : Using [0]eno1:10.9.3.241<0>
        rnd3:77634:79720 [0] NCCL INFO Using network Socket
        ```
        
        In other nodes, run with arg
      
        ```
        NCCL_SOCKET_IFNAME=eno1    
        ```
        
        </details>


    + <details><summary><b>Install CMake from source</b></summary>
        
        ```
        version=3.23
        build=2 ## don't modify from here
        mkdir ~/temp
        cd ~/temp
        wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
        tar -xzvf cmake-$version.$build.tar.gz
        cd cmake-$version.$build/
        ./bootstrap
        make -j8
        sudo make install
        ```
        
        </details>
        
    + <details><summary><b>Install NCCL Backend (Distributed training)</b></summary>
        
        ```
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        sudo apt install libnccl2 libnccl-dev
        ```
        
        </details>
        
    + <details><summary><b>Install MXNet from source</b></summary>
        
        ```
        git clone --recursive --branch 1.9.1 https://github.com/apache/incubator-mxnet.git mxnet
        cd mxnet
        cp config/linux_gpu.cmake config.cmake
        rm -rf build
        mkdir -p build && cd build
        cmake -DUSE_CUDA=ON -DUSE_CUDNN=OFF -DUSE_MKL_IF_AVAILABLE=OFF -DUSE_MKLDNN=OFF -DUSE_OPENMP=OFF -DUSE_OPENCV=ON -DUSE_BLAS=open ..
        make -j32
        cd ../python
        pip install --user -e .
        ```
        
        </details>

  
    + <details><summary><b>Tensorflow could not load dynamic library 'cudart64_101.dll'</b></summary>
        For above example tensorflow would require CUDA 10.1, please switch to CUDA 10.1 or change tensorflow version which compatible with CUDA version, check here: https://www.tensorflow.org/install/source#gpu
        </details>


    + <details><summary><b>Fix Deepstream (6.2+) FFMPEG OpenCV installation</b></summary>
        Fix some errors about undefined reference & not found of libavcodec, libavutil, libvpx, ...
          
        ```
        apt-get install --reinstall --no-install-recommends -y libavcodec58 libavcodec-dev libavformat58 libavformat-dev libavutil56 libavutil-dev gstreamer1.0-libav
        apt install --reinstall gstreamer1.0-plugins-good
        apt install --reinstall libvpx6 libx264-155 libx265-179 libmpg123-0 libmpeg2-4 libmpeg2encpp-2.1-0
        gst-inspect-1.0 | grep 264
        rm ~/.cache/gstreamer-1.0/registry.x86_64.bin
        apt install --reinstall libx264-155
        apt-get install gstreamer1.0-libav
        apt-get install --reinstall gstreamer1.0-plugins-ugly
        ```
        
        </details>

    + <details><summary><b>Gstreamer pipeline to convert MP4-MP4 with re-encoding</b></summary>

        ```
        gst-launch-1.0 filesrc location="<path-to-input>" ! qtdemux ! video/x-h264 ! h264parse ! avdec_h264 ! videoconvert ! x264enc ! h264parse ! qtmux ! filesink location=<path-to-output>
        ```
        
        </details>
  
    + <details><summary><b>Gstreamer pipeline to convert RTSP-RTMP</b></summary>

        ```
        gst-launch-1.0 rtspsrc location='rtsp://<path-to-rtsp-input>' ! rtph264depay ! h264parse ! flvmux ! rtmpsink location='rtmp://rtmp://<path-to-rtmp-output>'
        ```
        
        </details>

    + <details><summary><b>Gstreamer pipeline to convert RTSP-RTMP with reducing resolution</b></summary>

        ```
        gst-launch-1.0 rtspsrc location='rtsp://<path-to-rtsp-input>' ! rtpbin ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! video/x-raw,width=640,height=640 ! x264enc ! h264parse ! flvmux streamable=true ! rtmpsink location='rtmp://<path-to-rtmp-output>'
        ```

        </details>  
