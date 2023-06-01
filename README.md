# Quick Notes

## Experiment 1

Metric Learning， 改变解码位置

Using Focal CTC Loss, 防止数据分布不均衡

ACE Loss, 顺序和频度问题, lambda=0.1

## Experiment 2

分段式的CTC，避免双行车牌中 首行字母的对齐问题

## Experiment 3

flatten (reshape) 之后，使用 TCN

Accuracy: 67.00%
Single Accuracy: 82.13%
Double Accuracy: 7.41%
Final Accuracy: 74.97%

在 Flatten 之前添加 LayerNorm

```python
x = tf.split(f_map, num_or_size_splits=2, axis=1)
x = tf.concat(x, axis=2)
# layer norm
x = LayerNormalization()(x)
x = tf.reshape(x, (-1, 128, 168))
```

Accuracy: 70.25%
Single Accuracy: 83.86%
Double Accuracy: 16.67%
Final Accuracy: 77.30%

直接对f_map进行reshape

```python
x = tf.reshape(f_map, (-1, 128, 168))
```

Accuracy: 64.50%
Single Accuracy: 73.51%
Double Accuracy: 29.01%
Final Accuracy: 82.69%

change to FasterNet

Accuracy: 61.38%
Single Accuracy: 71.32%
Double Accuracy: 22.22%
Final Accuracy: 73.07%

## Experiment 4

using segmentation framework, directly connect to the ctc

Accuracy: 60.75%
Single Accuracy: 67.40%
Double Accuracy: 34.57%
Final Accuracy: 71.47%

## Experiment 5

remove TCN model, only CNN, 先行预览版 for auto IT

减少不必要的结构, 改用 separable conv 连接 4x 的downsample

Accuracy: 86.12%
Single Accuracy: 90.87%
Double Accuracy: 67.88%
Final Accuracy: 88.45%

## Experiment 6

lraspp 连接 1/16 的 downsample 的 cnn

Accuracy: 87.62%
Single Accuracy: 90.87%
Double Accuracy: 75.15%
Final Accuracy: 89.53%

## Experiment 7

连接更多的浅层特征

Single Accuracy: 90.08%
Double Accuracy: 64.24%
Final Accuracy: 87.26%

## Experiment 8

回归最淳朴的结构

Accuracy: 88.88%
Single Accuracy: 91.18%
Double Accuracy: 80.00%
Final Accuracy: 90.34%

## Experiment 9

repeat the train of double train and single double train

Original Accuracy:      90.38%
S. LPR   Accuracy:      91.50%
D. LPR   Accuracy:      86.06%
Final    Accuracy:      91.98%

## Experiment 10

Focal CTC Loss(alpha=0.3, gamma=5.0), SGD optimizer

Original Accuracy:      91.75%
S. LPR   Accuracy:      91.34%
D. LPR   Accuracy:      93.33%
Final    Accuracy:      92.44%

## Experiment 11

Focal CTC Loss(alpha=0.5, gamma=5.0), SGD optimizer

Original Accuracy:      93.62%
S. LPR   Accuracy:      93.07%
D. LPR   Accuracy:      95.76%
Final    Accuracy:      94.10%

## Experiment 12

Focal CTC Loss(alpha=0.1, gamma=5.0), SGD optimizer

Original Accuracy:      93.00%
S. LPR   Accuracy:      92.91%
D. LPR   Accuracy:      93.33%
Final    Accuracy:      93.94%

## Conclusion

Choose alpha=0.5, gamma=5.0, SGD optimizer

on Test dataset

Original Accuracy:      96.12%
S. LPR   Accuracy:      96.49%
D. LPR   Accuracy:      94.44%
Final    Accuracy:      97.59%

on Real dataset

Original Accuracy:      93.62%
S. LPR   Accuracy:      93.07%
D. LPR   Accuracy:      95.76%
Final    Accuracy:      94.10%
