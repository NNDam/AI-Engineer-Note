# Kaldi ASR with Triton-inference-server
Phần này sẽ đề cập đến cách sử dụng Kaldi backend trong Triton
### 1. Build
- Build docker image
```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/Kaldi/SpeechRecognition
scripts/docker/build.sh
```
- Download mô hình sample LibriSpeech
```
scripts/docker/launch_download.sh
```
- Khởi chạy triton-kaldi-server với LibriSpeech
```
scripts/docker/launch_server.sh
```
### 2. Load custom model
Phần này mình sẽ tiến hành sử dụng triton để load customized model.
- Tạo thư mục mới tại thư mục làm việc hiện tại
```
models/infer_asr_kaldi_radio_v1/1
```
với ```infer_asr_kaldi_radio_v1``` là tên model của mình.
- Run triton tại thư mục hiện tại với MODE ```EXPLICIT```
```
docker run --rm -it \
   --gpus device=0 \
   --shm-size=1g \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   -p8005:8000 \
   -p8006:8001 \
   -p8007:8002 \
   --name trt_server_asr \
   -v $PWD/data:/data \
   -v $PWD/model-repo:/mnt/model-repo \
   -v $PWD/models:/models \
   triton_kaldi_server tritonserver --model-repo=/models --model-control-mode=explicit
```
trong đó ```$PWD/models``` là thư mục ta vừa tạo
- Sử dụng một screen khác copy ```libtriton_kaldi.so```
```
docker ps
docker exec -it <CONTAINER-ID> bash
cp /workspace/model-repo/kaldi_online/1/libtriton_kaldi.so /models/infer_asr_kaldi_radio_v1/
```
- Xây dựng cấu trúc thư mục như sau (nhớ sửa lại đường dẫn trong các file ```.conf``` cho đúng):
```
├── models
│   ├── infer_asr_kaldi_radio_v1
│   │   ├── 1
│   │   │   ├── conf
│   │   │   │   ├── ivector_extractor.conf
│   │   │   │   ├── mfcc.conf
│   │   │   │   ├── online.conf
│   │   │   │   ├── online_cmvn.conf
│   │   │   │   ├── splice.conf
│   │   │   ├── ivector_extractor
│   │   │   │   ├── final.dubm
│   │   │   │   ├── final.ie
│   │   │   │   ├── final.mat
│   │   │   │   ├── global_cmvn.stats
│   │   │   │   ├── online_cmvn.conf
│   │   │   │   ├── online_cmvn_iextractor
│   │   │   │   ├── splice_opts
│   │   │   ├── final.mdl
│   │   │   ├── global_cmvn.stats
│   │   │   ├── HCLG.fst
│   │   │   ├── words.txt
│   │   ├── config.pbtxt
│   │   ├── libtriton_kaldi.so
```
  Lưu ý: file ```/models/infer_asr_kaldi_radio_v1/1/global_cmvn.stats``` khác với file ```/models/infer_asr_kaldi_radio_v1/1/ivector_extractor/global_cmvn.stats```
- Load model lên triton bằng [gRPC API](../docs/model_management.md)
