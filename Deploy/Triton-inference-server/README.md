# AI-Engineer-Note

Tất cả những thứ liên quan đến Triton-inference-server
## Basic
- [1. Cài đặt triton-server và triton-client](docs/install.md)
    + [1.1. Các chế độ quản lý model (load/unload/reload)](docs/model_management.md)
- [2. Sơ lược về các backend trong Triton](docs/backend.md)
- [3. Cấu hình cơ bản khi deploy mô hình](docs/model_configuration.md)
- [4. Deploy mô hình sử dụng ONNX-runtime và Triton](docs/triton_onnx.md)
- [5. Deploy mô hình sử dụng TensorRT-runtime và Triton](docs/triton_tensorrt.md)
- [6. Deploy mô hình sử dụng Pytorch (TorchScript) và Triton](docs/triton_pytorch.md)
- [7. Model Batching](docs/model_batching.md)
- [8. Ensemble Model và pre/post processing](docs/model_ensemble.md)
## Advance
- [Sử dụng Performance Analyzer Tool](docs/perf_analyzer.md)
- [Optimizations](#)
    + [Tối ưu Pytorch backend](docs/optimization_pytorch.md)