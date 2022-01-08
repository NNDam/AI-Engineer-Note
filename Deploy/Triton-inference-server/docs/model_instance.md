# Model Instance

Khi muốn scale up quy mô hệ thông, ta muốn sử dụng nhiều instance của model tương ứng với 1 hoặc nhiều GPU để tối đa hóa tốc độ, giảm thiểu độ trễ phía người dùng. Nghĩa là một requests từ phía người dùng có thể có nhiều lựa chọn hơn, khắc phục hiện tượng bottleneck phía inference. Do vậy, phần này mình sẽ trình bày về cấu hình Model Instance của ```triton-server```.
### 1. Cơ chế Model Instance trong triton-server
Kiến trúc Triton cho phép nhiều model và một hoặc nhiều instance của cùng một model thực thi song song trên hệ thống. Hệ thống có thể không có, có một hoặc nhiều GPU. Hình dưới đây minh họa với 2 model, giả sử Triton hiện không xử lý bất kỳ yêu cầu nào, khi 2 requests đến đồng thời, 1 request cho mỗi 1 model, Triton ngay lập tức lên lịch cho cả 2 requests trên GPU và thực hiện song song chúng. Nếu hệ thống không có GPU, lập lịch trên CPU thì sẽ tiến hành trên các luồng và phụ thuộc vào OS hệ thống.
<p align="left">
  <img src="../fig/multi_model_exec.png" width="800">
</p>

Mặc định, nếu nhiều requests đến cùng 1 model tại 1 thời điểm, Triton sẽ lập lịch sao cho chỉ xử lý 1 request mỗi một thời điểm
<p align="left">
  <img src="../fig/multi_model_serial_exec.png" width="800">
</p>

Triton cung cấp một config cho model được gọi là **instance-group** chỉ định số lượng executions được thực thi song song, mỗi execution như vậy được gọi là **instance**. Mặc định, Triton sẽ khởi tạo các **instance** trên các GPU khác nhau. Ví dụ như trong hình dưới đây, có 3 instances và 4 requests được gọi đến, request thứ 4 phải đợi cho đến khi 1 trong 3 lần thực thi đầu tiên hoàn thành trước khi bắt đầu.
<p align="left">
  <img src="../fig/multi_model_parallel_exec.png" width="800">
</p>