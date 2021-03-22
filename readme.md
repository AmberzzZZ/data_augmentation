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
        mixup: 全图叠加，类别加权, default alpha=1.0, 论文实验alpha=[0.1,0.2,0.4,4,8], alpha的值越大生成的lam偏向0.5的可能性更高
        cutout: 局部区域填0，类别不变
        cutMix: 局部区域填充其他图像，类别加权
    检测：
        mosaic: 4张图，边缘处理
    ref1: https://blog.csdn.net/weixin_38715903/article/details/103999227
    ref2: https://zhuanlan.zhihu.com/p/174019699


4. AutoAugment & RandAugment
    official: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py

    blend: (1-factor)\*image1 + factor\*image2，factor=0.的时候是image1，factor=1.的时候是image2

    cutout: 将固定大小的区域replace成指定像素值

    solarize: 阈值取反

    solarize_add: 阈值偏移

    color: 等价于PIL.ImageEnhance.Color，调整饱和度，factor=0.的时候是灰度图，factor越大饱和度越高

    posterize: 等价于PIL.ImageOps.posterize，每个颜色通道上，变量bits对应指定n个低bits置0

    equalize: 等价于PIL.ImageOps.equalize，产生灰度值均匀分布的图像

    wrap: 给image添加一个全1的channel，utils，用来保存图像形变的mask

    unwarp: 基于wrap那一维的mask，将0的地方对应的rgb通道填灰(128)，utils，用来补全rotate、translate、shear导致的图像空洞

    \*\_level_to_arg: 将0-10的magnitude映射到每个operation的实际参数

    distort_image_with_autoaugment: 把best policies用在image上，返回augmented versions

    distort_image_with_randaugment: 
        number_layers: N，其实是number of operations per sub-policy，choose in [1,3]
        magnitude: M，choose in [5, 30]
        返回一张随机增强的图





    




