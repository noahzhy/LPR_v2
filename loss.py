import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer
from itertools import groupby


# class CenterLossLayer(Layer):
#     def __init__(self, alpha=0.5, num_classes=100, feat_dim=128, name=None, **kwargs):
#         super(CenterLossLayer, self).__init__(name=name, **kwargs)
#         self.alpha = alpha
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim

#         self.init_centers()

#     def get_config(self):
#         config = super(CenterLossLayer, self).get_config()
#         config.update({
#             "alpha": self.alpha,
#             "num_classes": self.num_classes,
#             "feat_dim": self.feat_dim,
#             # "centers": self.centers
#         })
#         return config

#     def init_centers(self):
#         def init_centers():
#             return tf.zeros([self.num_classes, self.feat_dim])

#         self.centers = tf.Variable(
#             initial_value=init_centers,
#             dtype=tf.float32,
#             name='centers'
#         )

#     def call(self, features, predicts, y_true):
#         feats_reshape = tf.reshape(features, [-1, self.feat_dim])
#         print('feats_reshape:', feats_reshape.shape)
#         print('predicts:', predicts.shape)
#         label = tf.argmax(predicts, axis=-1)
#         print('label:', label.shape)
#         label = tf.reshape(label, [-1])
#         print('label:', label.shape)
#         batch_size = tf.shape(feats_reshape)[0]
#         quit()

#         # Calc L2 distance between feats and centers
#         square_feat = tf.reduce_sum(tf.square(feats_reshape), axis=1, keepdims=True)
#         square_feat = tf.tile(square_feat, [1, self.num_classes])

#         square_center = tf.reduce_sum(tf.square(self.centers), axis=1, keepdims=True)
#         square_center = tf.tile(square_center, [1, batch_size])
#         square_center = tf.transpose(square_center)

#         distmat = square_feat + square_center - 2.0 * tf.matmul(feats_reshape, tf.transpose(self.centers))

#         # Generate the mask
#         classes = tf.range(self.num_classes, dtype=tf.int64)
#         label = tf.expand_dims(label, axis=1)
#         label = tf.tile(label, [1, self.num_classes])
#         mask = tf.equal(classes, label)
#         mask = tf.cast(mask, tf.float32)

#         dist = tf.multiply(distmat, mask)

#         loss = tf.reduce_sum(tf.clip_by_value(dist, clip_value_min=1e-12, clip_value_max=1e+12)) / tf.cast(batch_size, tf.float32)
#         return loss


# tf.compat.v1.disable_eager_execution()

class CenterLossLayer(Layer):
    def __init__(self, alpha=0.5, num_classes=100, feat_dim=128, bs=1, name=None, **kwargs):
        super(CenterLossLayer, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.bs = bs

        self.init_centers()

    def get_config(self):
        config = super(CenterLossLayer, self).get_config()
        config.update({
            "alpha": self.alpha,
            "num_classes": self.num_classes,
            "feat_dim": self.feat_dim,
            # "centers": self.centers
        })
        return config

    def init_centers(self):
        def init_centers():
            return tf.zeros([self.num_classes, self.feat_dim])

        self.centers = tf.Variable(
            initial_value=init_centers,
            dtype=tf.float32,
            name='centers'
        )

        bs = 1

        # self.char_num = tf.compat.v1.placeholder(tf.int32, shape=[None,], name="char_num")
        # self.bat_labels = tf.compat.v1.placeholder(tf.int32, shape=[None,], name="bat_labels")
        # self.char_pos_init = tf.compat.v1.placeholder(tf.int32, shape=[None, 2], name='char_pos')

        # to variables
        # self.char_num = tf.Variable(tf.zeros([bs,], dtype=tf.int32), name='char_num')
        # self.bat_labels = tf.Variable(tf.zeros([bs,], dtype=tf.int32), name='bat_labels')
        # self.char_pos_init = tf.Variable(tf.zeros([bs, 2], dtype=tf.int32), name='char_pos')

    def call(self, y_pred, cnn_out, y_true):
        bs, f, c = tf.shape(cnn_out)[0], tf.shape(cnn_out)[1], tf.shape(cnn_out)[2]
        batch_size = K.shape(y_pred)[0]
        print('bs, f, c:', bs, f, c)
        print('batch_size:', batch_size)
        quit()
        # bs = 1

        # # single char labels required by center_loss op
        # self.bat_labels = tf.compat.v1.placeholder(tf.int32, shape=[None], name="bat_labels")
        self.bat_labels = y_true
        # # nums of chars in each sample, used to filter sample to do center loss
        # # self.char_num = tf.compat.v1.placeholder(tf.int32, shape=[None], name="char_num")
        self.char_num = tf.Variable(
            initial_value=tf.zeros([bs,], dtype=tf.int32),
            dtype=tf.int32,
            name='char_num')
        # # char pos: the positions of chars
        # # 因为 tensorflow 对在循环中长度改变的张量会报错，所以在此作为 placeholder 传入
        # # self.char_pos_init = tf.compat.v1.placeholder(tf.int32, shape=[None, 2], name='char_pos')
        self.char_pos_init = tf.Variable(tf.zeros([bs, 2], dtype=tf.int32), name='char_pos')

        # Reshape to the shape lstm needed. [batch_size, max_time, ..]
        # [batch_size, steps, features]
        self.embedding = cnn_reshaped
        self.raw_pred = tf.argmax(y_pred, axis=-1, name='raw_prediction')

        # 使用单个样本的对齐策略，如果一个样本中有重复预测，则去重后参与 center_loss 计算，如果有漏字，则不参与 center_loss 计算
        # 生成参与 center loss 计算的 embedding features 和标签
        self.char_label, self.embedding = self.raw_pred_to_features(
            self.raw_pred,
            self.bat_labels,
            self.embedding,
            self.char_num,
            self.char_pos_init)

        # 计算 center loss
        loss = self.get_center_loss(self.embedding, self.char_label)
        return loss

    def get_center_loss(self, features, labels):
        """获取center loss及center的更新op
        Arguments:
            features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
            labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
            alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
            num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
        Return:
            loss: Tensor,可与softmax loss相加作为总的loss进行优化.
            centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
        """
        # 获取特征的维数，例如256维
        len_features = features.get_shape()[1]

        # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心
        # update centers when training
        # centers = tf.Variable(tf.zeros([self.num_classes, len_features]), dtype=tf.float32, name='centers')
        centers = self.centers
        # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
        # labels = tf.reshape(labels, [-1])
        print('tf.shape(labels):', tf.shape(labels))

        # 构建label
        # 根据样本label,获取mini-batch中每一个样本对应的中心值
        centers_batch = tf.gather(centers, labels)
        # 计算loss
        loss = tf.nn.l2_loss(features - centers_batch)
        # 当前mini-batch的特征值与它们对应的中心值之间的差
        diff = centers_batch - features

        # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = self.alpha * diff

        centers_update_op = tf.compat.v1.scatter_sub(centers, labels, diff)

        with tf.control_dependencies([centers_update_op]):
            loss = tf.identity(loss)

        return loss

    @tf.function
    def get_char_pos_and_label(self, preds, label, char_num, poses):
        """
        过滤掉预测漏字的样本，返回过滤后的字符位置和标签
        Args:
            preds: 去掉重复字符后的预测结果，是字符的位置为 True, 否则为 False
            label: 字符标签
            char_num: 每个样本的字符数
            poses: 初始化的字符位置
        Returns:
            字符位置: 2D tensor of shape (num of chars, 2)，后一个维度为（字符位置，图片序号）
            标签: 1D tensor of shape (num of chars,)
        """
        i = tf.constant(0, dtype=tf.int32)
        char_total = tf.constant(0, dtype=tf.int32)

        for char in preds:
            char_pos = tf.cast(tf.where(char), tf.int32)

            # 判断预测出的字符数和 gt 是否一致，如果不一致则忽略此样本
            char_seg_num = tf.shape(char_pos)[0]

            if not tf.equal(char_seg_num, char_num[i]):
                # tf.print('切出的字符数量与真实值不同，忽略此样本')
                label = tf.concat([label[:char_total], label[char_total + char_num[i]:]], axis=0)
                i = tf.add(i, 1)
                continue
            else:
                char_total = tf.add(char_total, char_num[i])

            # 在seg中添加 batch 序号标识，方便后续获取 feature
            batch_i = char_pos[:, :1]
            batch_i = tf.broadcast_to(i, tf.shape(batch_i))
            char_pos = tf.concat([char_pos, batch_i], axis=1, name='add_batch_index')

            # 连接在一个 segs tensor 上
            poses = tf.concat([poses, char_pos], axis=0, name='push_in_segs')
            i = tf.add(i, 1)

        return poses[1:], label

    @staticmethod
    def get_features(char_pos, embedding):
        """
        根据字符的位置从相应时间步中获取 features
        Args:
            char_pos: 字符位置, 2D tensor of shape (num of chars, 2)，最后一个维度为（字符位置，图片序号）
            embedding: 输入全连接层的 tensor
        Returns:
            features: 字符对应的 feature
        """
        def get_slice(pos):
            feature_one_char = embedding[pos[1], pos[0], :]
            return feature_one_char

        features = tf.map_fn(get_slice, char_pos, dtype=tf.float32)
        return features

    def raw_pred_to_features(self, raw_pred, label, embedding, char_num, poses):
        """
        得到用于计算 centerloss 的 embedding features, 和对应的标签
        Args:
            raw_pred: 原始的预测结果，形如 [[6941, 6941, 0, 6941, 6941, 5, 6941], …]
            label: 字符标签，形如 [0,5,102,10,…]
            embedding: 全连接的输入张量
            char_num: 每个样本的字符数，用于校验是否可以对齐
            poses: 初始化的字符位置
        Returns:
            self.embedding: embedding features
            self.char_label: 和 embedding features 对应的标签
            self.char_pos: 和 embedding features 对应的字符位置
        """
        with tf.compat.v1.name_scope('pos'):
            # 判断是否为预测的字符
            is_char = tf.less(raw_pred, self.num_classes - 1)
            # 错位比较法，找到重复字符
            char_rep = tf.equal(raw_pred[:, :-1], raw_pred[:, 1:])
            tail = tf.greater(raw_pred[:, :1], self.num_classes - 1)
            char_rep = tf.concat([char_rep, tail], axis=1)
            # 去掉重复字符之后的字符位置，重复字符取其 [最后一次]出现的位置
            char_no_rep = tf.math.logical_and(is_char, tf.math.logical_not(char_rep))
            # 得到字符位置 和 相应的标签，如果某张图片预测出来的字符数量和gt不一致则跳过
            self.char_pos, self.char_label = self.get_char_pos_and_label(
                preds=char_no_rep,
                label=label,
                char_num=char_num,
                poses=poses)
            # 根据字符位置得到字符的 embedding
            self.embedding = self.get_features(self.char_pos, embedding)

        return self.char_label, self.embedding


class FocalLossLayer(Layer):
    def __init__(self, alpha=1.0, gamma=5.0, name=None, **kwargs):
        super(FocalLossLayer, self).__init__(name=name, **kwargs)
        self.loss_fn = K.ctc_batch_cost
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        input_length = K.tile([[K.shape(y_pred)[1]]], [K.shape(y_pred)[0], 1])
        label_length = K.tile([[K.shape(y_true)[1]]], [K.shape(y_true)[0], 1])
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        p = tf.exp(-loss)
        focal_ctc_loss = tf.multiply(tf.multiply(self.alpha, tf.pow((1 - p), self.gamma)), loss)
        return tf.reduce_mean(focal_ctc_loss)

    def get_config(self):
        config = super(FocalLossLayer, self).get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config


class CE_loss(Layer):
    def __init__(self, name="ce_loss"):
        super(CE_loss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, y_pred)
        return loss


class BCE_loss(Layer):
    def __init__(self, name="ce_loss"):
        super(BCE_loss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
        return loss


class BinaryFocalLoss(Layer):
    def __init__(self, alpha=0.5, gamma=2.0, name=None, **kwargs):
        super(BinaryFocalLoss, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def get_config(self):
        config = super(BinaryFocalLoss, self).get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        f_loss = self.alpha * (1 - y_pred) ** self.gamma * y_true * K.log(y_pred)  # β*(1-p̂)ᵞ*p*log(p̂)
        f_loss += (1 - self.alpha) * y_pred ** self.gamma * (1 - y_true) * K.log(1 - y_pred)  # (1-β)*p̂ᵞ*(1−p)*log(1−p̂)
        f_loss = -f_loss  # −[β*(1-p̂)ᵞ*p*log(p̂) + (1-β)*p̂ᵞ*(1−p)*log(1−p̂)]

        # Average over each data point/image in batch
        axis_to_reduce = range(1, K.ndim(f_loss))
        f_loss = K.mean(f_loss, axis=axis_to_reduce)
        return f_loss


class DiceLoss(Layer):
    def __init__(self, smooth=1e-5, name=None, **kwargs):
        super(DiceLoss, self).__init__(name=name, **kwargs)
        self.smooth = smooth

    def get_config(self):
        config = super(DiceLoss, self).get_config()
        config.update({"smooth": self.smooth})
        return config

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # flatten label and prediction tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2. * intersection + self.smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth)
        dice_loss = 1 - tf.reduce_mean(dice)
        return dice_loss
