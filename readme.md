1. pixel 
    RandomBrightness
    RandomContrast
    RandomHSV
    RandomLightingNoise: 改变通道顺序
    gamma

2. coords & scale
    RandomExpand: 扩充小目标, 均值填充
    RandomCrop: 随机遮挡目标


3. compose
    分类：
        mixup: 全图叠加，类别加权
        cutout: 局部区域填0，类别不变
        cutMix: 局部区域填充其他图像，类别加权
    检测：
        mosaic: 4张图，边缘处理
    ref1: https://blog.csdn.net/weixin_38715903/article/details/103999227
    ref2: https://zhuanlan.zhihu.com/p/174019699
    



