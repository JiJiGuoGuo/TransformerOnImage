#这个文件是作者的模型文件，#Gram Matrix能够反映该组向量中，各个向量之间的某种关系，self用来指向类的实例对象，而不是类本身
#tf.keras.layers.Layer是一个可调用对象，以张量作为输入和输出，计算方法定义在call()里，官方建议定义__init__()，build(self,input_hape)
#call(),get_config()
import tensorflow as tf
from model_modified import VisionTransformer
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#---------------------------------设置的超参数-----------------------------------
data_path = r'./Datasets/archive/fer2013.csv'
emotions = {'0':'anger','1':'disgust','2':'fear','3':'happy','4':'sad','5':'surprised','6':'nomral'}
logdir = 'logs'
image_size = 48
patch_size = 4
num_layers = 4
d_model = 64
num_heads = 8
mlp_dim = 128
lr = 3e-4
weight_decay = 1e-4
batch_size = 3
epochs = 50
num_classes = 7
dropout = 0.1
image_channel = 1
#----------------------------------设置超参数，结束-----------------------------------
AUTOTUNE = tf.data.experimental.AUTOTUNE
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
train_dataset = dataset.take(20)
test_dataset = dataset.skip(32297)

#shuffle操作
train_dataset = (train_dataset.cache().shuffle(5 * batch_size).batch(batch_size).prefetch(AUTOTUNE))

strategy = tf.distribute.MirroredStrategy()
#模型构建
print("----------------building model---------------------------")
with strategy.scope():  # 创建一个上下文管理器，能够使用当前的训练策略
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_classes=num_classes,  # 类别的数量
        d_model=d_model,  # 64
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        channels=image_channel,
        dropout=dropout,
    )
    # 用于配置训练方法，告知训练器使用的优化器，损失函数和准确率评测标准
    model.compile(
        # 交叉熵损失函数，from_logits=True会将结果转换成概率(softmax)，
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay),
        # 网络评价指标，如accuracy,sparse_accuracy,sparse_categorical_accuracy
        metrics=["accuracy"],)

#最小二乘拟合
print("-------------------start fitting-----------------------------")
history = model.fit(
    x =train_dataset,
    epochs=50,
    callbacks=[TensorBoard(log_dir=logdir, profile_batch=0),],)

x  = tf.random.uniform((batch_size,image_size,image_size,image_channel))
print(x.shape)
transformer = Transformer(vector_size,num_heads)
y = transformer(x)