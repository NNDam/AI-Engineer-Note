## Multi-head Attention Block

### 1. Expland
### 2. Pytorch Implementation
```
import torch
import torch.nn as nn
import torch.nn.functional as F

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    
class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, head_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert(in_channels % head_size == 0), 'The size of head should be divided by the number of channels.'

        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(in_channels)

        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.num = 0

    def forward(self, x, y=None):
        h_ = x
        h_ = self.norm1(h_)
        if y is None:
            y = h_
        else:
            y = self.norm2(y)

        q = self.q(y)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b, self.head_size, self.att_size ,h*w) 
        q = q.permute(0, 3, 1, 2) # b, hw, head, att

        k = k.reshape(b, self.head_size, self.att_size ,h*w) 
        k = k.permute(0, 3, 1, 2)

        v = v.reshape(b, self.head_size, self.att_size ,h*w) 
        v = v.permute(0, 3, 1, 2)


        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2,3)

        scale = int(self.att_size)**(-0.5)
        q.mul_(scale)
        w_ = torch.matmul(q, k)
        w_ = F.softmax(w_, dim=3)

        w_ = w_.matmul(v)

        w_ = w_.transpose(1, 2).contiguous() # [b, h*w, head, att]
        w_ = w_.view(b, h, w, -1)
        w_ = w_.permute(0, 3, 1, 2)

        w_ = self.proj_out(w_)

        return x+w_
```