#Reference:https://github.com/piergiaj/tgm-icml19

import torch.nn as nn
import torch
import torch.nn.functional as F
import temporal_structure_filter as tsf
import math

def compute_pad(stride, k, s):
    if s % stride == 0:
        return max(k - stride, 0)
    else:
        return max(k - (s % stride), 0)
    
class Attention(nn.Module):
    def __init__(self, d_key=1024, drop_ratio=0.2, causal=False):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value):
        dot_products = torch.matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * 1e10
            if key.is_cuda:
                tri = tri.cuda(key.get_device())
            dot_products.data.sub_(tri.unsqueeze(0))
        result=torch.matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)
        return result

class TGM(tsf.TSF):

    def __init__(self,num_f, length, c_in, c_out, soft=False,flow=False):
        super(TGM, self).__init__(num_f)        #temporal_structure_filter num_f== Ni

        self.length = length
        self.c_in = c_in
        self.c_out = c_out
        self.soft = soft
        self.flow=flow
        self.soft_attn = nn.Parameter(torch.Tensor(c_out * c_in, num_f))

        self.attention=Attention()

        # learn c_out combinations of the c_in channels
        if self.c_in > 1 and not soft:
            self.convs = nn.ModuleList([nn.Conv2d(self.c_in, 1, (1, 1)) for c in range(self.c_out)])

        if self.c_in > 1 and soft:
            self.cls_attn = nn.Parameter(torch.Tensor(1, self.c_out, self.c_in, 1, 1))

        self.linear2 = nn.Linear(2048, 1024)
        self.linear1 = nn.Linear(1024, 2048)


    def forward(self, x):
        #temporal structure filterを用いてカーネルを取得
        k = super(TGM, self).get_filters(torch.tanh(self.gamma), torch.tanh(self.center), self.length, self.length)     #torch.Size([N,L])

        #(C_out*C_in x N)*(NxL) を C_out*C_in x L　に変えるためにsoft attentionを適用
        #ガウシアンの数の方向に対して、アテンションの総和を作成
        soft_attn = F.softmax(self.soft_attn, dim=1)        #torch.Size([N,L]) ==>  torch.Size([Cin*Cout,N])

        #soft-attentionとカーネルkをかける
        k = torch.mm(soft_attn, k)      #torch.Size([Cin*Cout,N]) × torch.Size([N,L])  = torch.Size([Cin*Cout,L])
        k = k.unsqueeze(1).unsqueeze(1)         #torch.Size([Cin*Cout,L]) ==>  torch.Size([Cin*Cout,1,1,L])

        #  length
        t=x.size(2)
        p = compute_pad(1, self.length, t)
        pad_f = p // 2
        pad_b = p - pad_f
        x = F.pad(x, (pad_f, pad_b))        #torch.Size([C,D,T])

        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1:
            x = x.expand(-1, self.c_in, -1, -1)        

        # use groups to separate the class channels
        # but apply it in groups of C_out
        chnls = []
        for i in range(self.c_out):
            # gives k of C_in x1x1xL
            # output of C_in xDxT

            r=F.conv2d(x, k[i * self.c_in:(i + 1) * self.c_in], groups=self.c_in)

            r =r.squeeze(1)
            # 1x1 conv to combine C_in to 1

            #self.c_inが「1」ならば、c_in>1はFalseとなる
            if self.c_in > 1 and not self.soft:
                if not self.flow:
                    r = F.relu(self.convs[i](r)).squeeze(1)     #1*1の畳み込みを適用して、relu関数を使用
                else:
                    r=torch.max(input=r,dim=1)[0]

            chnls.append(r)

        # get C_out x DxT
        f = torch.stack(chnls, dim=1)

        if self.c_in > 1 and self.soft:
            a = F.softmax(self.cls_attn, dim=2).expand(f.size(0), -1, -1, f.size(3), f.size(4))
            f = torch.sum(a * f, dim=1)

        return f


class Kernel(tsf.TSF):
    def __init__(self,num_f, length):
        super(Kernel, self).__init__(num_f)
        self.length = length

    def forward(self):
        #temporal structure filterを用いてカーネルを取得
        k = super(Kernel, self).get_filters(torch.tanh(self.gamma), torch.tanh(self.center), self.length, self.length)     #torch.Size([N,L])
        return k

#提案されていたモデル
class TGMModel(nn.Module):
    def __init__(self):
        super(TGMModel, self).__init__()

        self.dropout = nn.Dropout(p=0.1)

        self.sub_event = TGM(num_f=16, length=5, c_in=1, c_out=4, soft=False,flow=False)
        self.sub_event_rgb2 = TGM(num_f=16, length=5, c_in=4, c_out=4, soft=False,flow=False)
        self.sub_event_flow2 = TGM(num_f=16, length=5, c_in=4, c_out=4, soft=False,flow=True)

    def forward(self, rgb, opticalflow):
        size=rgb.size()[2]
        # 1層目
        sub_event_1 = self.sub_event(rgb)     
        sub_event_2 = self.sub_event(opticalflow)
        sub_event_3 = self.sub_event(rgb+opticalflow)

        # 2層目
        sub_event_1 = self.sub_event_rgb2(sub_event_1)
        sub_event_2 = self.sub_event_flow2(sub_event_2)
        sub_event_3 = self.sub_event_rgb2(sub_event_1+sub_event_2+sub_event_3)

        # 3層目
        sub_event_1 = self.sub_event_rgb2(sub_event_1)
        sub_event_2 = self.sub_event_flow2(sub_event_2)
        sub_event_3 = self.sub_event_rgb2(sub_event_1 + sub_event_2 + sub_event_3)

        # ドロップアウト
        sub_event_1 = self.dropout(torch.max(sub_event_1, dim=1)[0])
        sub_event_2 = self.dropout(torch.max(sub_event_2, dim=1)[0])
        sub_event_3 = self.dropout(torch.max(sub_event_3, dim=1)[0])

        sub_event_1=torch.swapaxes(sub_event_1,1,2) 
        sub_event_2=torch.swapaxes(sub_event_2,1,2) 
        sub_event_3=torch.swapaxes(sub_event_3,1,2)

        h = nn.Conv1d(size, 341, 1).to('cuda')
        h2 = nn.Conv1d(size, 342, 1).to('cuda')

        # 1D畳み込み
        sub_event_1 = F.relu(h(sub_event_1)) 
        sub_event_2 = F.relu(h(sub_event_2))
        sub_event_3 = F.relu(h2(sub_event_3))
        cat_list = [sub_event_1, sub_event_2, sub_event_3]

        # 連結
        return torch.cat(cat_list, dim=1)