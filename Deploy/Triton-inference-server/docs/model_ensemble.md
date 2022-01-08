# Ensemble multiple models and pre/post-processing

Phần này mình sẽ trình bày nội dung liên quan đến Model Ensemble trong việc giải quyết 2 tình huống
- Xây dựng pipeline end-to-end khi kết hợp 2 hoặc nhiều model với nhau (output của model này là input của model khác)
- Tích hợp tiền xử lý / hậu xử lý vào pipeline

<i>Lưu ý: Cách giải quyết của 2 tình huống này là giống nhau</i>
### 1. Đặt vấn đề
Ví dụ như trong trường hợp của mình, mình sử dụng mô hình GFPGan với nhiều bộ dữ liệu khác nhau, từ đó có các phiên bản khác nhau của mô hình, các phiên bản này đều có đặc điểm chung là sử dụng **cùng** một phương pháp **tiền xử lý (pre-processing)** và **hậu xử lý (post-processing)**. Cách thức deploy hiện tại là đặt tiền/hậu xử lý ở phía ```client```, nhưng điều này sẽ khá bất cập khi scalable. Do vậy câu hỏi đặt ra là làm thế nào để tích hợp 2 thứ này vào triton một cách nhanh chóng và linh hoạt nhất để giảm thiểu chi phí chuyển giao trung gian và số lượng requests gửi đến. Triton có hỗ trợ chúng ta dưới dạng **Model Ensemble**. Ý tưởng chủ yếu được gói gọn trong 2 gạch đầu dòng sau:
- Quá trình tiền/hậu xử lý được build thành 1 model triton
- Tạo model ensemble: ```pre-processing -> infer -> post-processing```. Model này không phải là một model thực sự mà là một ```dataflow``` được xây dựng dựa trên model configuration
### 2. Convert model tiền/hậu xử lý
Lấy ví dụ việc tiền xử lý ảnh của mình như sau (sử dụng numpy & opencv-python, thuần CPU)
```
def triton_preprocess(cropped_face):
    rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)     # BGR sang RGB
    rgb = rgb.astype("float32") / 255.0                     # Rescale về đoạn [0, 1]
    rgb = (rgb - 0.5)/0.5                                   # Rescale từ [0, 1] về [-1, 1]
    rgb = np.expand_dims(rgb, axis = 0)                     # [256, 256, 3] -> [1, 256, 256, 3]
    return np.transpose(rgb, (0, 3, 1, 2))                  # [1, 256, 256, 3] -> [1, 3, 256, 256]

def triton_postprocess(net_out, min_max = (-1, 1)):
    net_out = np.transpose(net_out, (0, 2, 3, 1))                      # [1, 3, 256, 256] -> [1, 256, 256, 3]
    net_out = np.clip(net_out[0], min_max[0], min_max[1])              # [1, 256, 256, 3] -> [256, 256, 3] & clip
    net_out = (net_out - min_max[0]) / (min_max[1] - min_max[0])       # Rescale từ [-1, 1] về [0, 1]
    net_out = np.array(net_out * 255.0, dtype = np.uint8)              # Rescale từ [0, 1] về [0, 255] với uint8
    return cv2.cvtColor(net_out, cv2.COLOR_RGB2BGR)                    # RGB sang BGR
```
Tiến hành convert sang pytorch
```
class GFPGanPreprocessor(nn.Module):
    def __init__(self):
        super(GFPGanPreprocessor, self).__init__()
    def forward(self, x):
        x = x[:, :, [2, 1, 0]]                    
        x = x / 255.0
        x = (x - 0.5)/0.5
        x = torch.unsqueeze(x, 0)
        return torch.permute(x, (0, 3, 1, 2))

class GFPGanPostprocessor(nn.Module):
    def __init__(self):
        super(GFPGanPostprocessor, self).__init__()
    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))              
        x = torch.clamp(x, -1.0, 1.0)
        x = ((x + 1.0)/2.0*255.0).byte()
        return x[:, :, [2, 1, 0]] 
```
Sử dụng pytorch JIT, nếu mọi người chưa biết về JIT có thể xem qua bài viết này
- [Deploy mô hình sử dụng Pytorch (TorchScript) và Triton](./triton_pytorch.md)
```
# JIT
pre_model = GFPGanPreprocessor()
post_model = GFPGanPostprocessor()
pre_model.eval()
post_model.eval()

pre_x = torch.rand((256, 256, 3))
pre_traced_cell = torch.jit.trace(pre_model, (pre_x,), strict=False, check_trace=True)
print(pre_model(pre_x))
print(pre_traced_cell(pre_x))
pre_traced_cell.save('pre_traced_cell.pt')

post_x = torch.rand((1, 3, 256, 256))
post_traced_cell = torch.jit.trace(post_model, (post_x,), strict=False, check_trace=True)
print(post_model(post_x))
print(post_traced_cell(post_x))
post_traced_cell.save('post_traced_cell.pt')
```
Kết quả chúng ta thu được 2 file ```pre_traced_cell.pt``` và ```post_traced_cell.pt``` là 2 model pre/post-process
### 3. Đẩy model lên triton-server
Bước này khá là cơ bản, mình tiến hành đẩy 2 model lên triton với các cấu hình tương ứng sau
- Pre-process
```
name: "pre_gfpgan"
platform: "pytorch_libtorch"
max_batch_size: 0
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
    dims: [1, 3, -1, -1]
  }
]
```
- Post-process
```
name: "post_gfpgan"
platform: "pytorch_libtorch"
max_batch_size: 0
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [1, 3, -1, -1]
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
- Đẩy lên triton, nên sử dụng EXPLICIT MODE như trong hướng dẫn sau:
    + [Các chế độ quản lý model (load/unload/reload)](./model_management.md)

### 4. Tạo Ensemble Model
Thiết lập mô hình ensemble với input là ```raw_image```, output là ```image_out```
- Trong quá trình tiền xử lý, ```raw_image``` là input đầu vào ```input__0``` của model ```pre_gfpgan``` ta vừa load lên triton ở bước trên
- Model ```pre_gfpgan``` trả về ```preprocessed_image``` lại feed tương ứng vào ```input__0``` của model ```infer_face_restoration_v2.1```
- Output của model ```infer_face_restoration_v2.1``` ta đặt là ```net_out``` lại là input của model ```post_gfpgan``` - - Cuối cùng trả ra output của ```post_gfpgan``` là ```image_out``` đồng thời là output cuối cùng của model
```
name: "ens_face_restoration_v2.1"
platform: "ensemble"
max_batch_size: 0
input [
  {
    name: "raw_image"
    data_type: TYPE_UINT8
    dims: [-1, -1, 3]
  }
]
output [
  {
    name: "image_out"
    data_type: TYPE_UINT8
    dims: [-1, -1, 3]
  }
]
ensemble_scheduling {
  step [
    {
      model_name: "pre_gfpgan"
      model_version: -1
      input_map {
        key: "input__0"
        value: "raw_image"
      }
      output_map {
        key: "output__0"
        value: "preprocessed_image"
      }
    },
    {
      model_name: "infer_face_restoration_v2.1"
      model_version: -1
      input_map {
        key: "input__0"
        value: "preprocessed_image"
      }
      output_map {
        key: "output__0"
        value: "net_out"
      }
    },
    {
      model_name: "post_gfpgan"
      model_version: -1
      input_map {
        key: "input__0"
        value: "net_out"
      }
      output_map {
        key: "output__0"
        value: "image_out"
      }
    }
  ]
}
```

Thiết lập xong cấu hình, ta khởi tạo thư mục ```1``` **rỗng** để tạo phiên bản đầu tiên và đẩy lên triton-server là done