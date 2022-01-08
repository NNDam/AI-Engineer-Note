# Model Batching

Phần này ta sẽ tìm hiểu về một số cơ chế batching hỗ trợ bởi Triton

### Dynamic Batching
Dynamic Batching thì không cần đề cập nhiều, luồng các messages vào đồng thời sẽ được gom lại và infer theo batch, phương pháp này chủ yếu nhằm tăng [throughput](../docs/perf_analyzer.md) (dẫn đến tăng [latency](../docs/perf_analyzer.md) khi trong cùng một điều kiện về resources)
```
dynamic_batching { }
```
hoặc thêm cấu hình thời gian tối đa queue chờ messages mới (microseconds)
```
dynamic_batching {
    max_queue_delay_microseconds: 100
  }
```

### Ragged Batching
