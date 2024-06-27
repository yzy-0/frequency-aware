import torch
import numpy as np


def apply_dep_filter(depth_map_tensor, filter_type='high_pass', cutoff_freq=0.7):
    # 转换为灰度张量
    gray_depth_map = torch.mean(depth_map_tensor, dim=1, keepdim=True).cuda()

    # 傅里叶变换
    f_transform = torch.fft.fft2(gray_depth_map, dim=(-2, -1)).cuda()
    f_shift = torch.fft.fftshift(f_transform, dim=(-2, -1)).cuda()

    rows, cols = gray_depth_map.shape[-2:]
    crow, ccol = rows // 2, cols // 2

    # 生成频域滤波器
    mask = torch.zeros((1, 1, rows, cols), dtype=torch.float32).cuda()
    if filter_type == 'high_pass':
        mask[..., crow - int(cutoff_freq * crow):crow + int(cutoff_freq * crow),
        ccol - int(cutoff_freq * ccol):ccol + int(cutoff_freq * ccol)] = 1
    elif filter_type == 'low_pass':
        mask[..., crow - int(cutoff_freq * crow):crow + int(cutoff_freq * crow),
        ccol - int(cutoff_freq * ccol):ccol + int(cutoff_freq * ccol)] = 0
        mask = 1 - mask

    # 应用滤波器
    f_shift = f_shift * mask

    # 逆傅里叶变换
    f_ishift = torch.fft.ifftshift(f_shift, dim=(-2, -1)).cuda()
    img_back = torch.fft.ifft2(f_ishift, dim=(-2, -1)).cuda()
    img_back = torch.abs(img_back).cuda()

    return img_back


# 创建深度图张量
# depth_map_tensor = torch.randn(1, 3, 480, 640)  # 假设深度图大小为 480x640
#
# # 应用频域高通滤波器增强深度图像特征
# enhanced_depth_map_tensor = apply_dep_filter(depth_map_tensor, filter_type='high_pass', cutoff_freq=0.7)
# print(enhanced_depth_map_tensor.shape)
