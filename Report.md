# Report

## Introduction

这里放一堆废话，介绍一下这个项目是干什么的，为什么要做这个项目，这个项目的目标是什么，这个项目的意义是什么，这个项目的应用场景是什么，这个项目的应用前景是什么。

## Improvement

改进点（顺序需要调整）

1. 通过合成数据，合成了一些缺失的数据，尤其对于双行车牌的意义是重大的。
2. 优化了车牌识别算法，这里特指在双行车牌的识别上，提高了车牌识别的准确率。
3. 针对特定加速平台，优化了车牌识别算法，使得车牌识别的速度得到了提升。
本来还能补一个 Green channel 的性能提升的，但是这个项目用的是 Gray channel，所以就不补了。

### Synthesize Data

这里介绍一下合成数据的方法，以及合成数据的意义。贴一下合成数据的图片。还有统计一下整个数据集中，合成数据占比多少。Finally, we synthesize xxx images for double-row license plates. 贴一下全部数据集的字符分布图。

### Improve License Plate Recognition

这里介绍一下车牌识别算法的改进，以及改进的意义。贴一下车牌识别算法的准确率对比图。Finally, we improve the accuracy of license plate recognition from xxx to xxx.

Detail as follows:
    1. using mobilenetv3 small as backbone, 根据 apple 的论文，以及coral平台的 ops 针对移动端的计算资源，replace all hard-swish with relu6, and remove the SE module. 并且对后两个 neck 进行了去除，提高了运行效率。
    2. using focal ctc loss 替换了原来的 ctc loss，以便于更好的训练难以识别的字符。
    3. 随着近些年 Multi-Task Learning 的发展，我们将segmentation 和 车牌识别进行联合训练，降低了训练难度，提高了识别准确率。

### Accelerate License Plate Recognition

这里介绍一下针对特定加速平台的优化，以及优化的意义。贴一下车牌识别算法的速度对比图。Finally, we accelerate the speed of license plate recognition from xxx to xxx.

主要的速度提升来源于，特定化的加速平台，以及针对性的优化。这里可以贴一下加速平台的介绍，以及加速平台的优化方法。

## Experiment

这里补充几个实验，以及实验的结果。消融实验什么的，可以放在这里。
Some tips:

1. The accuracy of TCN improves from xxx to xxx, its better than LSTM.
2. The accuracy of focal ctc loss improves from xxx to xxx, its better than ctc loss.
3. The accuracy of segmentation improves from xxx to xxx, its better than xxx.
4. The synthesis data improves the accuracy of double-row license plates from xxx to xxx.

## Conclusion

这里放一堆废话，总结一下这个项目的意义，以及这个项目的应用前景。
