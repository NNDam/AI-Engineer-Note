# Pytorch with Triton-inference-server
Nội dung chính của phần này sẽ đề cập đến việc deploy model sử dụng Pytorch backend lên Triton-inference-server. Ở đây mình sẽ minh họa sử dụng mô hình GFPGan trong khôi phục khuôn mặt. Do cho đến thời điểm hiện tại mô hình GFPGan này không support việc convert sang ONNX cũng như TensorRT nên phải sử dụng Pytorch backend để deploy. Đối với các mô hình phổ biến thông thường mọi người có thể bỏ qua một số bước và **đi thẳng đến bước Sử dụng Pytorch JIT** để convert mô hình sang TorchScript. TorchScript là dạng lưu trữ cả trọng số (weights) và kiến trúc mô hình, tương tự như Tensorflow saved-model, TensorRT hay ONNX.

Mình sẽ minh họa bằng mô hình GFPGan đề cập ở trên. Mô hình này có sử dụng **customized layer** là ```FusedLeakyRELU``` và ```UpFirDn2d``` implement sử dụng C++. Tuy nhiên, **TorchScript chỉ hỗ trợ các toán tử được customized sử dụng Python**. Nếu như mô hình của các bạn không sử dụng toán tử hay layer nào bên ngoài thì có thể bỏ qua bước này, nếu có ta phải implement lại các toán tử này sử dụng Pytorch

#### 1. Convert các toán tử customized từ C++ sang Python (<i>Optional</i>)
Mình sẽ tiến hành implement lại toán tử FusedLeakyRELU bằng cách sử dụng LeakyRELU có sẵn trong Pytorch (lưu ý là phiên bản này của mình có sử dụng thêm ```bias``` và ```scale```)
```
class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale
        self.op = torch.nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, input):
        # Simple Implement
        b, c, w, h = input.shape
        # sample input: [B, 32, 256, 256]
        # sample bias: [32] -> Need to tile to [B, 32, 256, 256]
        bias = torch.tile(self.bias, (b,w,h,1))
        bias = bias.permute(0,3,1,2)
        x = self.op(torch.add(input, bias))
        x = x*self.scale
        return x
```
Tiếp đến là toán tử UpFirDn2d, hên là có phiên bản implement sẵn với input tương tự như mình cần  
```
def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :, ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)
```
#### 2. Sử dụng Pytorch JIT
Tiếp đến chúng ta sẽ sử dụng đến ```torch.jit.trace``` để convert mô hình Pytorch sang TorchScript, lưu ý rằng ```torch.jit.trace``` sẽ không record bất cứ dạng điều kiện if-else hay vòng lặp for-while nào không cố định. Nếu gặp lỗi trong quá trình convert thì các bạn sẽ phải tự xây dựng lại đoạn đó (không thì thôi, các bạn có thể nhảy đến sử dụng JIT luôn). Ví dụ như mình sửa hàm forward trong mô hình để loại bỏ các trường hợp đó như sau:
- Hàm ```forward``` mặc định của mình bao gồm khá nhiều tham số, nhưng khi inference chỉ sử dụng đến một số tham số nhất định và giá trị cũng là cố định
```
DEFINITION
    forward (self,
            x,
            return_latents=False,
            save_feat_path=None,
            load_feat_path=None,
            return_rgb=True,
            randomize_noise=True)
CALL INFERENCE
    output = model(x, return_rgb=False, randomize_noise=False)
```
- Hàm ```forward``` trước khi sửa:
```
def forward(self,
                x,
                return_latents=False,
                save_feat_path=None,
                load_feat_path=None,
                return_rgb=True,
                randomize_noise=True):
        conditions = []
        unet_skips = []
        out_rgbs = []

        feat = self.conv_body_first(x)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)

        feat = self.final_conv(feat)

        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](feat)
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        if save_feat_path is not None:
            torch.save(conditions, save_feat_path)
        if load_feat_path is not None:
            conditions = torch.load(load_feat_path)
            conditions = [v.cuda() for v in conditions]
        image, _ = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=return_latents,
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=randomize_noise)

        return image, out_rgbs
```
- Hàm ```forward``` sau khi sửa:
```
def forward(self, x):
        conditions = []
        unet_skips = []
        # out_rgbs = [] <--- Unused, comment
        feat = self.conv_body_first(x)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        feat = self.final_conv(feat)
        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)
        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](feat)
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())
            # if return_rgb:  <--- Conditions, comment
            #     out_rgbs.append(self.toRGB[i](feat))
            
        # if save_feat_path is not None: <--- Conditions, comment
        #     torch.save(conditions, save_feat_path)
        # if load_feat_path is not None: <--- Conditions, comment
        #     conditions = torch.load(load_feat_path)
        #     conditions = [v.cuda() for v in conditions]

        # decoder
        image, _ = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=False, # <--- Fixed to False
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=False)  # <--- Fixed to False

        # return image, out_rgbs
        return image
```
Tiến hành JIT, với cùng một input thì kết quả trước và sau khi JIT **phải giống nhau**
```
x = torch.rand((1, 3, 256, 256)).to(device)
traced_cell = torch.jit.trace(model, (x,), strict=False, check_trace=True)
print(model(x))
print(traced_cell(x))
traced_cell.save('traced_model.pt')
```
Kết quả như sau
```
<Output-of-original-model>
tensor([[[[0.4016, 0.4957, 0.5073,  ..., 0.5910, 0.5702, 0.5630],
          [0.4932, 0.5128, 0.5017,  ..., 0.5684, 0.5582, 0.5986],
          [0.4990, 0.5119, 0.5058,  ..., 0.5688, 0.5585, 0.5648],
          ...,
          [0.4555, 0.4555, 0.4518,  ..., 0.5441, 0.5452, 0.5688],
          [0.4548, 0.4722, 0.4678,  ..., 0.5585, 0.5615, 0.5697],
          [0.4241, 0.4640, 0.4822,  ..., 0.5789, 0.5960, 0.4753]],

         [[0.3977, 0.5019, 0.5022,  ..., 0.4755, 0.4720, 0.4407],
          [0.4981, 0.5206, 0.5069,  ..., 0.4605, 0.4648, 0.4976],
          [0.4808, 0.5134, 0.5064,  ..., 0.4626, 0.4696, 0.4642],
          ...,
          [0.5199, 0.4985, 0.5021,  ..., 0.5059, 0.5035, 0.5388],
          [0.5017, 0.4992, 0.5063,  ..., 0.5123, 0.5174, 0.5325],
          [0.4148, 0.5015, 0.5192,  ..., 0.5553, 0.5606, 0.4201]],

         [[0.4313, 0.5356, 0.5719,  ..., 0.5755, 0.5607, 0.5146],
          [0.5305, 0.5412, 0.5459,  ..., 0.5069, 0.5000, 0.5563],
          [0.5338, 0.5405, 0.5534,  ..., 0.5292, 0.5166, 0.5303],
          ...,
          [0.5378, 0.5200, 0.5379,  ..., 0.5053, 0.5008, 0.5221],
          [0.5347, 0.5270, 0.5398,  ..., 0.5089, 0.5166, 0.5177],
          [0.4609, 0.5421, 0.5806,  ..., 0.5616, 0.5572, 0.4033]]]],
       device='cuda:0', grad_fn=<AddBackward0>)
       
<Output-of-jitted-model>
tensor([[[[0.4016, 0.4957, 0.5073,  ..., 0.5910, 0.5702, 0.5630],
          [0.4932, 0.5128, 0.5017,  ..., 0.5684, 0.5582, 0.5986],
          [0.4990, 0.5119, 0.5058,  ..., 0.5688, 0.5585, 0.5648],
          ...,
          [0.4555, 0.4555, 0.4518,  ..., 0.5441, 0.5452, 0.5688],
          [0.4548, 0.4722, 0.4678,  ..., 0.5585, 0.5615, 0.5697],
          [0.4241, 0.4640, 0.4822,  ..., 0.5789, 0.5960, 0.4753]],

         [[0.3977, 0.5019, 0.5022,  ..., 0.4755, 0.4720, 0.4407],
          [0.4981, 0.5206, 0.5069,  ..., 0.4605, 0.4648, 0.4976],
          [0.4808, 0.5134, 0.5064,  ..., 0.4626, 0.4696, 0.4642],
          ...,
          [0.5199, 0.4985, 0.5021,  ..., 0.5059, 0.5035, 0.5388],
          [0.5017, 0.4992, 0.5063,  ..., 0.5123, 0.5174, 0.5325],
          [0.4148, 0.5015, 0.5192,  ..., 0.5553, 0.5606, 0.4201]],

         [[0.4313, 0.5356, 0.5719,  ..., 0.5755, 0.5607, 0.5146],
          [0.5305, 0.5412, 0.5459,  ..., 0.5069, 0.5000, 0.5563],
          [0.5338, 0.5405, 0.5534,  ..., 0.5292, 0.5166, 0.5303],
          ...,
          [0.5378, 0.5200, 0.5379,  ..., 0.5053, 0.5008, 0.5221],
          [0.5347, 0.5270, 0.5398,  ..., 0.5089, 0.5166, 0.5177],
          [0.4609, 0.5421, 0.5806,  ..., 0.5616, 0.5572, 0.4033]]]],
       device='cuda:0', grad_fn=<AddBackward0>)
With rtol=1e-05 and atol=1e-05, found 171 element(s) (out of 196608) whose difference(s) exceeded the margin of error (including 0 nan comparisons). The greatest difference was 0.00031512975692749023 (0.5294076800346375 vs. 0.52909255027771), which occurred at index (0, 1, 230, 138).
```
Kết quả cho thấy sự sai khác tối đa là **0.00031512975692749023** không đáng kể, như vậy việc convert sang TorchScript của chúng ta là thành công.
#### 3. Xây dựng cấu hình cho model
Bước cuối cùng đơn giản là xây dựng cấu hình ```config.pbtxt``` cho model và đẩy vào triton-inference-server. Cấu trúc thư mục tương tự như sau:
```bash
├── models
│   ├── wav2vec_general_v1        # <--- TensorRT
│   │   ├── 1
│   │   │   ├── model.plan
│   │   ├── config.pbtxt
│   ├── wav2vec_general_v2        # <--- ONNX-Runtime
│   │   ├── 1
│   │   │   ├── model.onnx
│   │   ├── config.pbtxt
│   ├── gfpgan                    # <--- TorchScript
│   │   ├── 1
│   │   │   ├── model.pt (file traced_model.pt đã convert phía trên)
│   │   ├── config.pbtxt
```
trong đó thư mục ```models``` là thư mục mount đến của Docker triton-inference-server như mình trình bày trong cài đặt triton. Thư mục con là các model và cấu hình tương ứng, với thứ tự bất kỳ và engine bất kỳ mà triton hỗ trợ. Tuy nhiên lưu ý rằng đối với TorchScript thì tên biến phải được quy chuẩn dạng ```input__{}``` và ```output__{}```, chẳng hạn của mình như sau:
```
name: "gfpgan"
platform: "pytorch_libtorch"
max_batch_size: 0
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [1, 3, 256, 256]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [1, 3, 256, 256]
  }
]
```
Ở đây input đầu vào của mình fix cứng dạng **static batch**, hay **static shape** [1, 3, 256, 256], mình có thể cấu hình dạng **dynamic batch** với xử lý tối đa 8 inputs một lúc như sau:
```
name: "gfpgan"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [3, 256, 256]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [3, 256, 256]
  }
]
```

Tham khảo thêm về cách cấu hình Batch tại
- [Model Configuration](../docs/model_configuration.md)
- [Model Batching](../docs/model_batching.md)

<i>(Việc gọi gRPC đến triton bằng python là giống nhau với mọi engine, cho nên mình sẽ không trình bày lại)</i>