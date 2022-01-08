# Model Management

Có 3 chế độ quản lý model trong triton đó là **NONE** (mặc định), **EXPLICIT** (động) và **POLL**

### NONE Mode (Default)
- Cấu hình ```--model-control-mode=none```
- Triton sẽ tiến hành load toàn bộ mô hình cùng cấu hình tương ứng lên bộ nhớ, những model nào bị lỗi sẽ bỏ qua và không khả dụng.
- Việc thay đổi repo của model khi server đang chạy sẽ không tác động đến hệ thống hiện tại
- **Không thể** sử dụng ```load``` và ```unload``` API từ ```triton-client```
- Ưu điểm:
    + Dễ sử dụng
- Nhược điểm:
    + Khó tùy biến
    + Việc bổ sung/loại bỏ models đòi hỏi **phải** khởi động lại ```triton-server```
### EXPLICIT Mode (Recommend)
- Cấu hình ```--model-control-mode=explicit```
- Mặc định triton sẽ **không** ```load``` model nào vào bộ nhớ nếu flag ```--load-model``` không được khai báo. Do vậy, với khởi động mặc định cần phải call API ```load``` các model cần thiết **bằng tay**
- Các model có thể được gọi ```load``` và ```unload``` tùy ý thông qua API từ ```triton-client```
- Việc thay đổi repo của model khi server đang chạy sẽ tác động đến hệ thống hiện tại: **load lại model đó**
- Ưu điểm:
    + Dễ tùy biến
    + Việc bổ sung/loại bỏ models **không cần** khởi động lại ```triton-server```
- Nhược điểm:
    + Hơi khó để làm quen và sử dụng

Tham khảo API ```Load/Unload/Reload``` model sử dụng Python tại [đây](../src/sample_load_unload.py)
### POLL
Thấy bảo là không recommend trong **production** nên cũng lười không đọc luôn ...