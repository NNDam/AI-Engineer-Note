# AI-Engineer-Note

Tất cả những thứ liên quan đến Triton-inference-server
## Basic
- [1. Cài đặt triton-server và triton-client](docs/install.md)
    + [1.1. Các chế độ quản lý model (load/unload/reload)](docs/model_management.md)
- [2. Sơ lược về các backend trong Triton](docs/backend.md)
- [3. Cấu hình cơ bản khi deploy mô hình](docs/model_configuration.md)
- [4. Deploy mô hình](#)
    - [4.1 ONNX-runtime](docs/triton_onnx.md)
    - [4.2 TensorRT](docs/triton_tensorrt.md)
    - [4.3 Pytorch & TorchScript](docs/triton_pytorch.md)
    - [4.4 Kaldi <i>(Advanced)</i>](docs/triton_kaldi.md)
- [5. Model Batching](docs/model_batching.md)
- [6. Ensemble Model và pre/post processing](docs/model_ensemble.md)
## Advanced
- [Sử dụng Performance Analyzer Tool](docs/perf_analyzer.md)
- [Optimizations](#)
    + [Tối ưu Pytorch backend](docs/optimization_pytorch.md)
