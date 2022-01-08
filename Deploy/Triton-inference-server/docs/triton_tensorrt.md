# TensorRT-runtime with Triton-inference-server

Trong trường hợp muốn deploy mô hình sử dụng TensorRT-runtime thay vì ONNX-runtime (model thường phải convert sang ONNX trước khi sang TensorRT), file weight **phải** được convert theo **đúng** phiên bản TensorRT mà docker triton-inference-server đang sử dụng. Do vậy, ta truy cập vào môi trường docker hiện tại như sau:

- Lấy ID của Docker đang chạy triton-inference-server
```
damnguyen@rnd3:~$ docker ps

CONTAINER ID   IMAGE                                   COMMAND                  CREATED        STATUS        PORTS                                                           NAMES
6ef0b4972292   nvcr.io/nvidia/tritonserver:21.12-py3   "/opt/tritonserver/n…"   23 hours ago   Up 23 hours   0.0.0.0:8000-8002->8000-8002/tcp, :::8000-8002->8000-8002/tcp   cranky_hamilton
b09d98350935   quay.io/cloudhut/kowl:master-645e3b4    "./kowl"                 6 days ago     Up 6 days                                                                     gifted_davinci
```
ta có CONTAINER ID của triton là ```6ef0b4972292```
- Chạy bash sử dụng triton container
```
damnguyen@rnd3:~$ docker exec -it 6ef0b4972292 bash
root@6ef0b4972292:/opt/tritonserver#
```
- Convert model ONNX sang TensorRT (cú pháp tương tự khi làm việc với engine TensorRT thông thường)
```
/usr/src/tensorrt/bin/trtexec --onnx=<path-to-onnx> --saveEngine=<path-to-save-plan-file>
```
- Deploy model lên triton tương tự như ONNX sử dụng tên platform là ```tensorrt_plan``` thay vì ```onnxruntime_onnx```
    + [Deploy mô hình sử dụng ONNX-runtime và Triton](./triton_onnx.md)