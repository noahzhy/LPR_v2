# Report of LPR

@Haoyu, Zhang

Date: 2023/05/24

描述：针对 LPR 的改进



## 摘要

1. 优化了车牌识别的算法，针对双行车牌的识别准确率有显著提升，并且改用 Focal CTC Loss 以减少韩文字符和数字分布不均对模型训练的影响。
2. 对于新的硬件平台 coral 进行了针对性的优化了车牌识别算法，使得车牌识别的速度得到了提升。
3. 利用合成数据，针对车牌检测和车牌识别缺少的双行车牌的数据进行了补充。并对于一行车牌的数据进行扩展，尤其是在电动车牌和新式车牌部分进行了补足。



## 内容

## 算法改进

近些年来随着多任务学习的发展，越来越多的实验证明多任务学习能够进一步提高准确度。与单任务学习相比，主要有以下几个方面的优势。

1. 多任务学习通过挖掘任务之间的关系，能够得到额外的有用信息，大部分情况下都要比单任务学习的效果要好。在有标签样本比较少的情况下，单任务学习模型往往不能够学习得到足够的信息，表现较差，多任务学习能克服当前任务样本较少的缺点，从其他任务里获取有用信息，学习得到效果更好、更鲁棒的机器学习模型
2. 多任务学习有更好的模型泛化能力，通过同时学习多个相关的任务，得到的共享模型能够直接应用到将来的某个相关联的任务上。

受此启发，针对 LPR 的任务设计了如下的网络。

<img src="E:\projects\LPR_v2\report_img\tinyLPR.svg" alt="tinyLPR" style="zoom: 80%;" />



针对同一张输入的车牌，进行图像分割和车牌识别两个任务的多任务学习。模型在训练时，同时生成分割的结果以及车牌结果，但在实际部署的推理模型中，我们去除了负责分割任务的head以进一步精简模型和提高推理速度。

首先对输入的图像进行简单的图像处理，得到大致的字符掩码。由于数据量较大，且图像分割的准确度并不是我们优先考虑的指标，所以仅仅使用简单的 OpenCV 的图像处理操作来生成对应的掩码。

<img src="E:\projects\LPR_v2\report_img\mask_0.jpg" alt="mask_0" style="zoom:50%;" />

通过将这些字符掩码同样作为输入，可以加快神经网络模型的收敛速度，并且提高模型精度。



车牌数据存在明显的区域性分布，并且不同字符的分布也存在着较大的差距。

![letters](E:\projects\LPR_v2\report_img\letters.png)其中，A-Q，a-q分别表示

```python
{
    '서울': 'A', '부산': 'B', '대구': 'C', '인천': 'D',
    '광주': 'E', '대전': 'F', '울산': 'G', '세종': 'H',
    '경기': 'I', '강원': 'J', '충북': 'K', '충남': 'L',
    '전북': 'M', '전남': 'N', '경북': 'O', '경남': 'P',
    '제주': 'Q'
}
```

因此，我们采用 Focal CTC Loss 来尽可能的避免字符分布不均存在的问题。

<img src="E:\projects\LPR_v2\report_img\fctc.png" alt="fctc" style="zoom: 80%;" />

结合公式和 LPR 的字符分布情况，alpha 取 1.0，gamma 取 5.0。







考虑到 LPR 任务的实时性要求，我们选择了 mobilenet v3 small 作为该网络的 backbone，并且针对 coral 硬件的特性，进行了针对性的优化。

1. 去除了原版 mobilenet v3 small 中包含的所有 SE-Net 模块。SE-Net 虽然可以提高模型的精度，但是SE-Net 将会增加近 40% 的延迟。

   ![se](E:\projects\LPR_v2\report_img\se.jpg)

2. 由于coral 对于 hard-swish 激活函数的支持较差，考虑到后续量化的算子支持。这里我们选择使用 relu6 替代原版 mobilenet v3 small 中的 hard-swish 激活函数。

   <img src="E:\projects\LPR_v2\report_img\hswish.png" alt="hswish" style="zoom: 25%;" />

3. 由于深度表征对于 LPR 这个任务的贡献不大，且带来了较大的推理延迟。所以去除了原版 mobilenet v3 small 的最后两个 bneck。

<img src="E:\projects\LPR_v2\report_img\MobileNetsv3-6.png" alt="MobileNetsv3-6" style="zoom: 67%;" />



## 针对硬件优化

针对 EdgeTPU 支持的算子优化神经网路的设计，如下图所示，目标平台100%支持该网络的所有算子。图中绿色表示支持的算子，红色为不支持的算子。右下方的红点表示的是 Focal CTC Loss 损失函数，该部分仅在训练时参与运算，部署推理时不参与运算。

<img src="E:\projects\LPR_v2\report_img\tb.jpg" alt="tb" style="zoom:80%;" />

相比较之前版本的，经过 uint8 量化后的模型在下述平台上的运行速度如下：

| Platform | Description              | Inference latency |
| -------- | ------------------------ | ----------------- |
| CPU      | rk3399 Cortex-A72@1.8GHz | 32 ms             |
| TPU      | Coral USB Accelerator    | 3 ms              |



## 合成数据

由于韩国现存车牌种类较多，且年度跨度较大。实际收集到的车牌数据存在明显的地域性，且分布不均。

<img src="E:\projects\LPR_v2\report_img\info_klp.png" alt="info_klp" style="zoom:50%;" />



为了解决这一问题，我们通过 Blender 这一软件进行车牌建模并生成大量符合实际情况的车牌数据。并且考虑了夜间可能存在的车牌架光照影响，针对性的进行了模拟并最终生成了大量的合成车牌数据集。

<img src="E:\projects\LPR_v2\report_img\blender.png" alt="blender" style="zoom: 33%;" />

<img src="D:\materials_of_synthetic_data\2023-03-17_17-22-15.png" alt="2023-03-17_17-22-15" style="zoom:50%;" />

合成数据集对模型的精度提升有着显著的贡献。