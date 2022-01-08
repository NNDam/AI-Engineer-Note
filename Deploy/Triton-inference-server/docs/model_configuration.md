# Model Configuration
Mặc định, cấu hình phải định nghĩa trước cho mô hình các thông số như tên model, platform sử dụng (```tensorrt_plan, pytorch_libtorch, tensorflow_savedmodel, ...```), kiểu dữ liệu, kích thước cho input, output, cấu hình wramup, cấu hình optimization, ...
### 1. Cấu hình cơ bản (minimal model configuration)
Mặc định ta không cần xây dựng cấu hình cho các model TensorRT, Tensorflow saved-model và ONNX vì Triton có thể tự động generate. Đối với các model này nếu như không tồn tại ```config.pbtxt``` và ta khởi động triton-server với tham số ```--strict-model-config = false```, triton-server sẽ tự động generate ra file ```config.pbtxt``` ở mức cơ bản. Hoặc ta có thể xây dựng file ```config.pbtxt``` bằng tay. Ở đây mình sẽ xây dựng cấu hình cho đoạn code Pre-processing, Inference và Post-processing GFPGan đều sử dụng Pytorch.
- Pre-processing
```
name: "pre_gfpgan_batch"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input__0"
    data_type: TYPE_UINT8
    dims: [-1, -1, 3]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [3, -1, -1]
  }
]
```
- Inference
```
name: "infer_face_restoration_v2.1"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [3, 512, 512]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [3, 512, 512]
  }
]
```
- Post-processing
```
name: "post_gfpgan_batch"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [3, -1, -1]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_UINT8
    dims: [-1, -1, 3] 
  }
]
```

<i>Giá trị **-1** thể hiện cho **dynamic-shape** </i>

Cần lưu ý giá trị ```max_batch_size```, khi giá trị này **khác** 0, giá trị ```dims``` sẽ được hiểu là kích thước của **1 dữ liệu đầu vào**, model sẽ chấp nhận kích thước đầu vào từ ```1 x dims``` đến ```max_batch_size x dims``` (dynamic batch), và nếu giá trị này **bằng** 0, giá trị ```dims``` sẽ được hiểu là **kích thước đầu vào** (static batch)