# Optimize Pytorch Backend
Trong quá trình khởi động ```triton-server``` để load các mô hình sử dụng ```pytorch``` backend đôi khi ta sẽ gặp những thông báo kiểu:
```
I1227 03:45:06.216251 1 libtorch.cc:1255] TRITONBACKEND_ModelInitialize: license_plate_restoration_square_v1.1 (version 1)
I1227 03:45:06.216786 1 libtorch.cc:251] Optimized execution is enabled for model instance 'license_plate_restoration_square_v1.1'
I1227 03:45:06.216796 1 libtorch.cc:269] Inference Mode is disabled for model instance 'license_plate_restoration_square_v1.1'
I1227 03:45:06.216800 1 libtorch.cc:344] NvFuser is not specified for model instance 'license_plate_restoration_square_v1.1'
```
Đây là thông báo khi **Inference Mode** và **NvFuser** chưa được bật để tối ưu tốc độ. Do vậy trong phần này mình sẽ trình bày về cấu hình tối ưu ```triton-server``` khi sử dụng ```pytorch``` backend với các tham số phù hợp.

### 1. Inference Mode

**InferenceMode** hoạt động tương tự như **NoGradMode** khi không sử dụng autograd. Do vậy, trong đại đa số trường hợp khi mà model của chúng ta không quá đặc biệt (chứa những toán tử bị ảnh hưởng bởi autograd) thì ta có thể bật **InferenceMode** trong file cấu hình như sau:

```
parameters: {
key: "INFERENCE_MODE"
    value: {
    string_value:"true"
    }
}
```

- Kết quả khi tắt **Inference Mode** (mặc định)
```
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 46.4 infer/sec, latency 24657 usec
Concurrency: 2, throughput: 53.8 infer/sec, latency 41444 usec
Concurrency: 3, throughput: 54 infer/sec, latency 59257 usec
Concurrency: 4, throughput: 53.4 infer/sec, latency 81955 usec
```
- Kết quả sau khi bật (được cải thiện một chút)
```
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 42.6 infer/sec, latency 27506 usec
Concurrency: 2, throughput: 54.4 infer/sec, latency 40857 usec
Concurrency: 3, throughput: 54 infer/sec, latency 60192 usec
Concurrency: 4, throughput: 53.6 infer/sec, latency 81830 usec
```

### 2. NvFuser (CUDA Graph Fuser)
Nếu như các bạn có đọc qua về **TensorRT Optimization** thì cơ chế của **NvFuser** sẽ tương tự. Đơn giản là sẽ tiến hành Fuse một số toán tử lại với nhau để tăng tốc độ thực thi. Cơ chế fusing này đã trở nên rất phổ biến và được tích hợp vào hầu hết các framework hiện nay.
Tiến hành bật **NvFuser**:
```
parameters: {
key: "ENABLE_NVFUSER"
    value: {
    string_value:"true"
    }
}
```

### 3. Các chế độ Optimization khác
Ngoài ra, ta có một số **optimization flags** khác có thể thử
```
ENABLE_JIT_EXECUTOR
```
```
ENABLE_JIT_PROFILING
```
```
ENABLE_TENSOR_FUSER
```
Lưu ý rằng việc enable toàn bộ các ```optimization flags``` chưa chắc đã mang lại kết quả tốt nhất. Khuyến nghị chỉ sử dụng **INFERENCE_MODE** làm mặc định. Dưới đây là kết quả khi enable tất cả các ```optimization flags```

```
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 42.2 infer/sec, latency 27052 usec
Concurrency: 2, throughput: 48.2 infer/sec, latency 46771 usec
Concurrency: 3, throughput: 49.8 infer/sec, latency 65506 usec
Concurrency: 4, throughput: 29 infer/sec, latency 189399 usec
```

### 4. Model Instance
Việc bật nhiều **instance** giúp chúng ta tăng tốc độ khi luồng requests đầu vào có nhiều lựa chọn hơn (nhiều consumers). Tuy nhiên trong một số trường hợp các ```optimization flags``` thường gây ra một số lỗi cho nên khi lựa chọn việc sử dụng nhiều **instance models** ta nên ```DISABLE_OPTIMIZED_EXECUTION```
```
parameters: {
key: "DISABLE_OPTIMIZED_EXECUTION"
    value: {
    string_value:"true"
    }
}
```