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

Accuracy: 94.50%
Single Accuracy: 94.66%
Double Accuracy: 93.75%
Final Accuracy: 96.31%
