# Collection of FAQ about CUDA & Linux & apt-packages

<details><summary><b>Build OpenCV from source</b></summary>

- [Build OpenCV from source](docs/build_opencv.md)

</details>

<details><summary><b>Install Math Kernel Library (MKL/BLAS/LAPACK/OPENBLAS)</b></summary>
You are recommended to install all Math Kernel Library and then compile framework (e.g pytorch, mxnet) from source using custom config for optimization.
For AMD CPU, we recommend to use OpenBLAS

```
sudo apt install libjpeg-dev libpng-dev libblas-dev libopenblas-dev libatlas-base-dev liblapack-dev liblapacke-dev gfortran 
```

For Intel CPU, of course we need to use MKL & MKLDNN

```
sudo apt install intel-mkl-full
```

</details>

<details><summary><b>Fresh install NVIDIA driver (PC/Laptop/Workstation)</b></summary>

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

<details><summary><b>Check current CUDA version</b></summary>

```
nvcc --version
```

</details>

<details><summary><b>Check current supported CUDA versions</b></summary>

```
ls /usr/local/
```

</details>

<details><summary><b>Select GPU devices</b></summary>

```
CUDA_VISIBLE_DEVICES=<index-of-devices> <command>
CUDA_VISIBLE_DEVICES=0 python abc.py
CUDA_VISIBLE_DEVICES=0 ./sample.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python abc.py
CUDA_VISIBLE_DEVICES=0,1,2,3 ./sample.sh
```

</details>

<details><summary><b>Switch CUDA version</b></summary>

```
CUDA_VER=11.3
export PATH="/usr/local/cuda-$CUDA_VER/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VER/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

</details>

<details><summary><b>Error with distributed training NCCL (got freezed)</b></summary>

```
export NCCL_P2P_DISABLE="1"
```

</details>

<details><summary><b>Install CMake from source</b></summary>

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

<details><summary><b>Install MXNet from source (for AMD CPU & NVIDIA GPU)</b></summary>

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

  
<details><summary><b>Tensorflow could not load dynamic library 'cudart64_101.dll'</b></summary>
For above example tensorflow would require CUDA 10.1, please switch to CUDA 10.1 or change tensorflow version which compatible with CUDA version, check here: https://www.tensorflow.org/install/source#gpu
</details>

### Computer Vision
<details><summary><b>Gstreamer pipeline to convert MP4-MP4</b></summary>

```
gst-launch-1.0 filesrc location="path-to-input" ! qtdemux ! video/x-h264 ! h264parse ! qtmux ! filesink location=<path-to-output>
```

</details>
  
<details><summary><b>Gstreamer pipeline to convert RTSP-RTMP</b></summary>

```
gst-launch-1.0 rtspsrc location='rtsp://<path-to-rtsp-input>' ! rtph264depay ! h264parse ! flvmux ! rtmpsink location='rtmp://rtmp://<path-to-rtmp-output>'
```

</details>

<details><summary><b>Gstreamer pipeline to convert RTSP-RTMP with reducing resolution</b></summary>

```
gst-launch-1.0 rtspsrc location='rtsp://<path-to-rtsp-input>' ! rtpbin ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! video/x-raw,width=640,height=640 ! x264enc ! h264parse ! flvmux streamable=true ! rtmpsink location='rtmp://<path-to-rtmp-output>'
```

</details>  
