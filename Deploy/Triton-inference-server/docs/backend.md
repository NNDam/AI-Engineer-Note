# Triton backend

Triton backend được xây dựng trong việc thực thi mô hình. Một backend thông thường có thể được wrap bằng việc sử dụng các deep-learning framework như Pytorch, Tensorflow, TensorRT, ONNX-runtime hoặc OpenVINO như chúng ta đã từng làm để deploy mô hình (chẳng hạn như việc xây dựng một class load mô hình, warmup, pre-processing, inference, post-processing, ...). Dựa trên ý tưởng như vậy, ```triton-backend``` cũng được xây dựng bằng việc tổng hợp các backend của các deep-learning framework trên, sau đó cung cấp ra ngoài những API để người dùng có thể kết nối tới các mô hình deep-learning đã được load bằng ```triton-server```. Cho đến phiên bản hiện tại, ```triton-server``` hỗ trợ các backend sau:
- TensorRT (platform: ```tensorrt_plan```)
- Pytorch (platform: ```pytorch_libtorch```)
- ONNX (platform: ```onnxruntime_onnx```)
- Tensorflow (platform: ```tensorflow_savedmodel```)
- Other backends (platform: phụ thuộc vào backend đã được định nghĩa)