import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = SeparableConvBNReLU(dim, dim, kernel_size=3)
        self.local2 = SeparableConvBN(dim, dim, kernel_size=3)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = x + self.local2(self.local1(x))  # 残差连接关闭1X1

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


class transformer_block(nn.Module):
    def __init__(self, dim:int=256, out_dim:int=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)  # 把输出维度扩大了两倍
        self.transconv = ConvBN(dim, out_dim, kernel_size=1, stride=1)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        att_out = self.norm3(x)
        x = x + self.drop_path(self.mlp(self.norm1(x)))
        x = self.norm2(x)
        xmlp = x
        x = self.transconv(x)

        return x, torch.cat([xmlp, att_out], dim=1)  # 输入下一层的x，mlp的x，自注意力后的x
class Upsample(nn.Module):
    def __init__(self, inchannels: int, outchannels: int):
        super().__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2, stride=2, padding=0),
                                nn.BatchNorm2d(outchannels),
                                nn.ReLU()
                                )

    def forward(self, x):
        return self.up(x)

class GatedConvolutions(nn.Module):
    def __init__(self, in_channels,out_channels, drop_path=0.):
        super(GatedConvolutions, self).__init__()
        self.convgate=SeparableConvBNReLU(in_channels,in_channels,kernel_size=3,)
        self.conv=SeparableConvBNReLU(in_channels,in_channels,kernel_size=3)

        # 定义门控卷积层
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.conv1x1=nn.Conv2d( in_channels,out_channels,kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out=self.convgate(x)
        out=self.relu(out)*self.sigmoid(out)
        x=self.conv(x)
        out= self.conv1x1(out+x)

        return self.drop_path(out)


class GDSCNet(nn.Module):
    def __init__(self, dim: list = [32, 64, 128, 256, 512], inputchannel: int = 3, classnum: int = 6,
                 num_heads: list = [4, 4, 16, 16, 16], mlp_ratio: list = [4, 4, 4, 4, 4],
                 drop_path_encoder: float = 0., drop_path_decoder: float = 0.):
        super().__init__()
        self.convdim = ConvBNReLU(inputchannel, dim[0])  # relu
        self.pool = nn.MaxPool2d(2)
        self.encoder1 = transformer_block(dim=dim[0], out_dim=dim[1], num_heads=num_heads[0], mlp_ratio=mlp_ratio[0],
                                          drop_path=drop_path_encoder)
        self.encoder2 = transformer_block(dim=dim[1], out_dim=dim[2], num_heads=num_heads[1], mlp_ratio=mlp_ratio[1],
                                          drop_path=drop_path_encoder)
        self.encoder3 = transformer_block(dim=dim[2], out_dim=dim[3], num_heads=num_heads[2], mlp_ratio=mlp_ratio[2],
                                          drop_path=drop_path_encoder)
        self.encoder4 = transformer_block(dim=dim[3], out_dim=dim[4], num_heads=num_heads[3], mlp_ratio=mlp_ratio[3],
                                          drop_path=drop_path_encoder)
        self.encoder5 = transformer_block(dim=dim[4], out_dim=dim[4], num_heads=num_heads[4], mlp_ratio=mlp_ratio[4],
                                          drop_path=drop_path_encoder)

        self.upsample5 = Upsample(dim[3], dim[3])
        self.decoder5 = GatedConvolutions(in_channels=dim[4] * 3, out_channels=dim[3], drop_path=0.)

        self.upsample4 = Upsample(dim[2], dim[2])
        self.decoder4 = GatedConvolutions(in_channels=dim[3] * 3, out_channels=dim[2], drop_path=0.)

        self.upsample3 = Upsample(dim[1], dim[1])
        self.decoder3 = GatedConvolutions(in_channels=dim[2] * 3, out_channels=dim[1], drop_path=0.)

        self.upsample2 = Upsample(dim[0], dim[0])
        self.decoder2 = GatedConvolutions(in_channels=dim[1] * 3, out_channels=dim[0], drop_path=0.)

        self.upsample1 = Upsample(dim[0], dim[0])
        self.decoder1 = GatedConvolutions(in_channels=dim[0] * 3, out_channels=dim[0], drop_path=0.)

        self.out_decoder = nn.Sequential(nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(dim[0]),
                                         nn.ReLU(),
                                         nn.Conv2d(dim[0], classnum, kernel_size=1, stride=1, padding=0)
                                         )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, out):
        out = self.convdim(out)
        out = self.pool(out)
        out, xsa1 = self.encoder1(out)

        out = self.pool(out)
        out, xsa2 = self.encoder2(out)

        out = self.pool(out)
        out, xsa3 = self.encoder3(out)

        out = self.pool(out)
        out, xsa4 = self.encoder4(out)

        out = self.pool(out)
        out, xsa5 = self.encoder5(out)

        out = torch.cat([xsa5, out], dim=1)
        out = self.decoder5(out)
        out = self.upsample5(out)

        out = torch.cat([xsa4, out], dim=1)
        out = self.decoder4(out)
        out = self.upsample4(out)

        out = torch.cat([xsa3, out], dim=1)
        out = self.decoder3(out)
        out = self.upsample3(out)

        out = torch.cat([xsa2, out], dim=1)
        out = self.decoder2(out)
        out = self.upsample2(out)

        out = torch.cat([xsa1, out], dim=1)
        out = self.decoder1(out)
        out = self.upsample1(out)

        return self.out_decoder(out)

if __name__ == "__main__":
    from thop import profile
    import torch

    model = GDSCNet().to("cuda")
    input = torch.randn(1, 3, 512, 512).to("cuda")
    flops, params = profile(model, inputs=(input,))

    print(f"FLOPS: {flops}, Params: {params}")

