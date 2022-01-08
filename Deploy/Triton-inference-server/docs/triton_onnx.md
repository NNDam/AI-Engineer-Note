# ONNX-runtime with Triton-inference-server

Để deploy ONNX model chạy với ONNX-runtime (ngoài ONNX-runtime có thể sử dụng TensorRT-runtime nếu support), ta cần để platform là ```onnxruntime_onnx```, ngoài ra các tham số cơ bản trong cấu hình cũng tương tự. Mình sẽ tiến hành deploy model ```wav2vec_general_v2``` như sau:
- Trong thư mục ```models```, khởi tạo thư mục ```wav2vec_general_v2``` chứa file cấu hình và weights
- Để file weights dưới đường dẫn ```models/wav2vec_general_v2/1/model.onnx```, trong đó ```1``` là phiên bản của mô hình
- Để file config dưới đường dẫn ```models/wav2vec_general_v2/config.pbtxt```, lưu ý không ném trong thư mục phiên bản

```
name: "wav2vec_general_v2"
platform: "onnxruntime_onnx"
max_batch_size : 0
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [1, -1]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, -1, 105]
  }
]
```
- Đẩy model lên triton-server
```
python src/sample_load_unload.py wav2vec_general_v2
```