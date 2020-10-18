#这个文件是作者的模型文件，#Gram Matrix能够反映该组向量中，各个向量之间的某种关系，self用来指向类的实例对象，而不是类本身
#tf.keras.layers.Layer是一个可调用对象，以张量作为输入和输出，计算方法定义在call()里，官方建议定义__init__()，build(self,input_hape)
#call(),get_config()
import tensorflow as tf
import os
import tensorflow_addons as tfa
from argparse import ArgumentParser
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model_modified import VisionTransformer
AUTOTUNE = tf.data.experimental.AUTOTUNE
data_path = r'./Datasets/archive/fer2013.csv'
emotions = {'0':'anger','1':'disgust','2':'fear','3':'happy','4':'sad','5':'surprised','6':'nomral'}

#将图片，标签打包到一个元组中
def pack_image_vector(features,labels):
    return features,labels
#tensorflow的csv文件读取，返回一个Dataset对象
def get_csv_file(csv_data_path):
    return tf.data.experimental.make_csv_dataset(
        csv_data_path,batch_size=3,select_columns=['emotion','pixels'],
        label_name='emotion',na_value="?",num_epochs=1,ignore_errors=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=48, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=8, type=int)  # 多头注意力机制数量
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)  # AdamW权重衰减系数
    parser.add_argument("--batch-size", default=3, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--num-classes", default=7, type=int)  # 分类总类别数
    parser.add_argument("--dropout", default=0.1, type=float)  # dropout概率
    parser.add_argument("--img-chan", default=1, type=int)  # 图像通道数
    args = parser.parse_args()
    np.set_printoptions(precision=3,suppress=True)

    # 读取csv文件
    df = pd.read_csv(filepath_or_buffer=data_path, usecols=["emotion", "pixels"], dtype={"pixels": str})
    fer_pixels = df.copy()

    # 分成特征和标签
    fer_label = fer_pixels.pop('emotion')
    fer_pixels = np.asarray(fer_pixels)

    # 将特征转换成模型需要的类型
    fer_train = []
    for i in range(len(fer_label)):
        pixels_new = np.asarray([float(p) for p in fer_pixels[i][0].split()]).reshape(48, 48, 1)
        fer_train.append(pixels_new)
    fer_train = np.asarray(fer_train)
    fer_label = np.asarray(fer_label)
    # 转换为tf.Dateset类型
    dataset = tf.data.Dataset.from_tensor_slices((fer_train, fer_label))

    #数据集验证集测试集的拆分
    train_dataset = dataset.take(28708)
    temp_dataset = dataset.skip(28708)
    val_dataset = temp_dataset.take(3589)
    test_dataset = temp_dataset.skip(3589)
    test_dataset = test_dataset.take(3589)

    #shuffle操作
    train_dataset = (dataset.cache().shuffle(5 * args.batch_size).batch(args.batch_size).prefetch(AUTOTUNE))
    strategy = tf.distribute.MirroredStrategy()
    #模型构建
    print("----------------building model---------------------------")
    with strategy.scope():  # 创建一个上下文管理器，能够使用当前的训练策略
        model = VisionTransformer(
            image_size=48,
            patch_size=4,
            num_layers=4,
            num_classes=7,  # 类别的数量
            d_model=64,  # 64
            num_heads=4,
            mlp_dim=128,
            channels=1,
            dropout=0.1,
        )
        # 用于配置训练方法，告知训练器使用的优化器，损失函数和准确率评测标准
    #     model.compile(
    #         # 交叉熵损失函数，from_logits=True会将结果转换成概率(softmax)，
    #         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #         optimizer=tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay),
    #         # 网络评价指标，如accuracy,sparse_accuracy,sparse_categorical_accuracy
    #         metrics=["accuracy"],)
    #
    # #最小二乘拟合
    # print("-------------------start fitting-----------------------------")
    # history = model.fit(
    #     x =train_dataset,
    #     validation_data=val_dataset,
    #     epochs=50,
    #     callbacks=[TensorBoard(log_dir=args.logdir, profile_batch=0),],)
    #
    # # tf.keras.Model.predict()
    # # model.predict()
    #
    # # os.path.join路径拼接
    # model.save_weights(os.path.join(args.logdir, "vit"))
    print(len(model.layers))

