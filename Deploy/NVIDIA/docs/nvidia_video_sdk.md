# FFMPEG hardware acceleration with Nvidia Video SDK
## 1. Requirements
- GPU with hardware-acceleration support
<p align="center">
  <img src="../fig/support_nvenc_nvdec.png" width="1080">
  <i>Example of NVDEC support</i>
</p>
Link: https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new

- Nvidia Driver
- CUDA Toolkit

## 2. Install FFMPEG with hardware acceleration
```
sudo apt-get install build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev libx264-dev

git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git

cd nv-codec-headers && sudo make install && cd ..

git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/ && cd ffmpeg

./configure --enable-nonfree --enable-cuda-nvcc --enable-nvenc --enable-cuvid --enable-nvdec --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared --enable-libx264 --enable-libvpx --enable-libvorbis --enable-gpl

make

sudo make install
```
- Check FFMPEG
```
ffmpeg --help
```
If you meet error like
```
ffmpeg: error while loading shared libraries: lib<abc>.so: cannot open shared object file: No such file or directory
```
try add this line to the end of ```/etc/ld.so.conf```:
```
/usr/local/lib
```
then ```sudo ldconfig```

## 3. Benchmark
### 3.1. Convert MPEG-4 to H264
- Public **libx264**
```
ffmpeg -y -i test.avi -c:v libx264 -preset veryfast test.mp4

[libx264 @ 0xaad740] using SAR=1/1
[libx264 @ 0xaad740] using cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2 AVX512
[libx264 @ 0xaad740] profile High, level 4.0
[libx264 @ 0xaad740] 264 - core 155 r2917 0a84d98 - H.264/MPEG-4 AVC codec - Copyleft 2003-2018 - http://www.videolan.org/x264.html - options: cabac=1 ref=1 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=2 psy=1 psy_rd=1.00:0.00 mixed_ref=0 me_range=16 chroma_me=1 trellis=0 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=0 threads=34 lookahead_threads=8 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=1 keyint=250 keyint_min=20 scenecut=40 intra_refresh=0 rc_lookahead=10 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
Output #0, mp4, to test.mp4:
  Metadata:
    software        : Lavf58.3.100
    encoder         : Lavf59.20.101
  Stream #0:0: Video: h264 (avc1 / 0x31637661), yuv420p(progressive), 1920x1080 [SAR 1:1 DAR 16:9], q=2-31, 20 fps, 10240 tbn
    Metadata:
      encoder         : Lavc59.25.100 libx264
    Side data:
      cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: N/A
frame= 3569 fps=141 q=-1.0 Lsize=  395781kB time=00:02:58.30 bitrate=18184.1kbits/s speed=7.03x
video:395734kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.011851%
```
- Hardware acceleration
```
ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -i test.avi -c:v h264_nvenc test.mp4

Input #0, avi, from 'test.avi':
  Metadata:
    software        : Lavf58.3.100
  Duration: 00:02:58.40, start: 0.000000, bitrate: 40980 kb/s
  Stream #0:0: Video: mpeg4 (Simple Profile) (XVID / 0x44495658), yuv420p, 1920x1080 [SAR 1:1 DAR 16:9], 40864 kb/s, 600 fps, 20 tbr, 600 tbn
Stream mapping:
  Stream #0:0 -> #0:0 (mpeg4 (native) -> h264 (h264_nvenc))
Press [q] to stop, [?] for help
Output #0, mp4, to '/home/damnguyen/Desktop/Dataset/Survelliance/capture_9_gpu.mp4':
  Metadata:
    software        : Lavf58.3.100
    encoder         : Lavf59.20.101
  Stream #0:0: Video: h264 (Main) (avc1 / 0x31637661), cuda(progressive), 1920x1080 [SAR 1:1 DAR 16:9], q=2-31, 2000 kb/s, 20 fps, 10240 tbn
    Metadata:
      encoder         : Lavc59.25.100 h264_nvenc
    Side data:
      cpb: bitrate max/min/avg: 0/0/2000000 buffer size: 4000000 vbv_delay: N/A
frame= 3569 fps=390 q=37.0 Lsize=   44851kB time=00:02:58.40 bitrate=2059.5kbits/s speed=19.5x
video:44836kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.034549%
```
So basically we can increase performance from **7.03x** to **19.5x** with NVIDIA hardware-acceleration