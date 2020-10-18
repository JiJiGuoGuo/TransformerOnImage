
import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import TensorBoard

from model_modified import VisionTransformer

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)#多头注意力机制数量
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)#AdamW权重衰减系数
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--num-classes",default=6,type=int)#分类总类别数
    parser.add_argument("--dropout",default=0.1,type=float)#dropout概率
    parser.add_argument("--img-chan",default=3,type=int)#图像通道数
    args = parser.parse_args()

#从官网，下载数据集，当download=True时，下载name的数据集，默认路径在用户目录下
    # ds = tfds.load("imagenet_resized/32x32", as_supervised=True)
    ds = tfds.load(name="cifar10",as_supervised=True,split=['train','test'])

    #数据预处理阶段，cache缓存数据，shuffle打乱数据，prefetch预先调入数据的元素，大多数的数据集的调用应以prefetch结束，用内存开销改善延迟
    ds_train = (ds[0].cache().shuffle(5 * args.batch_size).batch(args.batch_size).prefetch(AUTOTUNE))
    ds_test = (ds[1].cache().batch(args.batch_size).prefetch(AUTOTUNE))

#使用分布式训练，MirroredStrategy允许在一台机器的多个GPU上进行同步训练，如果没有找到GPU，则使用CPU,在进程中训练
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():#创建一个上下文管理器，能够使用当前的训练策略
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            num_classes=args.num_classes,#类别的数量
            d_model=args.d_model,#64
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            channels=args.img_chan,
            dropout=args.dropout,
        )
        #用于配置训练方法，告知训练器使用的优化器，损失函数和准确率评测标准
        model.compile(
            #交叉熵损失函数，from_logits=True会将结果转换成概率(softmax)，
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            optimizer=tfa.optimizers.AdamW(
                learning_rate=args.lr, weight_decay=args.weight_decay
            ),
            #网络评价指标，如accuracy,sparse_accuracy,sparse_categorical_accuracy
            metrics=["accuracy"],
        )

    #最小二乘拟合
    model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=args.epochs,
        callbacks=[TensorBoard(log_dir=args.logdir, profile_batch=0),],
    )
    #os.path.join路径拼接
    model.save_weights(os.path.join(args.logdir, "vit"))
