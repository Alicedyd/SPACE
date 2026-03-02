import os
import cv2
import numpy as np
from scipy.fftpack import dct, idct
import argparse
from pathlib import Path
from PIL import Image

def dct2d(image):
    """
    对图像进行2D DCT变换
    """
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def idct2d(dct_coeffs):
    """
    对DCT系数进行2D逆DCT变换
    """
    return idct(idct(dct_coeffs.T, norm='ortho').T, norm='ortho')

def frame_freq_mix(img1, img2, threshold_x, threshold_y):
    """
    处理一对图像：DCT变换 -> 频率域替换 -> 逆DCT变换
    
    Args:
        img1: 第一个PIL Image对象
        img2: 第二个PIL Image对象  
        threshold_x: x坐标阈值
        threshold_y: y坐标阈值
        
    Returns:
        PIL Image对象: 处理后的图像
    """
    # 将PIL Image转换为numpy数组
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    # 确保两个图像尺寸相同
    assert img1_array.shape == img2_array.shape, "输入的两张图像尺寸必须相同"
    
    # 转换为浮点型并归一化
    img1_array = img1_array.astype(np.float32) / 255.0
    img2_array = img2_array.astype(np.float32) / 255.0
    
    # 处理每个颜色通道
    result = np.zeros_like(img1_array)
    
    for channel in range(img1_array.shape[2]):
        # 对每个通道进行DCT变换
        dct1 = dct2d(img1_array[:, :, channel])
        dct2 = dct2d(img2_array[:, :, channel])
        
        # 创建结果DCT系数，初始为img1的DCT系数
        result_dct = dct1.copy()
        
        # 获取图像尺寸
        height, width = dct1.shape
        
        # 根据阈值条件替换频率域系数
        for y in range(height):
            for x in range(width):
                # 如果x坐标和y坐标大于对应阈值，则用img2的DCT系数替换
                if x > threshold_x and y > threshold_y:
                    result_dct[y, x] = dct2[y, x]
        
        # 逆DCT变换回图像域
        result[:, :, channel] = idct2d(result_dct)
    
    # 限制像素值范围并转换回uint8
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)
    
    # 转换回PIL Image对象
    result_image = Image.fromarray(result)
    
    return result_image

if __name__ == "__main__":
    img1 = Image.open("/root/autodl-tmp/AIGC_data/Kinetics/kinetics-dataset/k400/train_rec_crop/crop_frames/-__F5CLvIlo_000039_000049_000.png").convert("RGB")
    img2 = Image.open("/root/autodl-tmp/AIGC_data/Kinetics/kinetics-dataset/k400/train_rec_crop/rec_frames/-__F5CLvIlo_000039_000049_000.png").convert("RGB")

    result_image = process_image_pair(img1, img2, threshold_x=12.5, threshold_y=12.5)
    
    result_image.save("result_image.png")