import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
# model = tf.keras.models.Sequential()
# model.add(tf.keras.Input(shape=(16,)))
# # model.add(tf.keras.layers.Dense(32,activation='relu',use_bias=True))
# model.add(tf.keras.layers.Dense(32))
#
# print(model.output_shape)#@property用于将类方法直接转换为类属性


# def self_attention(size,x):
# #后面跟x表示上一层的输出，通过线性映射将x映射到QKV三个不同的矩阵（实际上是对应三个不同的权重矩阵）
#     key = tf.keras.layers.Dense(size)(x)
#     value = tf.keras.layers.Dense(size)(x)
#     query = tf.keras.layers.Dense(size)(x)
#
#     atten = tf.matmul(query,key,transpose_b=True)
#     print("shape key[-1]： {}".format(key.shape[-1]))
#     atten = atten/tf.sqrt(tf.cast(vector_size,tf.float32))
#     atten_left = tf.nn.softmax(atten)
#     atten_sum = tf.matmul(atten_left,value)
#     return atten_sum
#
# def muti_head_self_attention(x):
#     x_sep = tf.reshape(x,[batch_size,sequence_length,num_heads,vector_size // num_heads])
#     x_mh = tf.transpose(x_sep,[0,2,1,3])
#     y = self_attention(vector_size//num_heads,x_mh)
#     y = tf.transpose(y,[0,2,1,3])
#     y = tf.reshape(y,[batch_size,sequence_length,vector_size])
#     return y

class MultiHeadSelfAttention(tf.keras.Model):
    def __init__(self,heads=1,vector_size=32):
        print("--------------------初始化被执行")
        super(MultiHeadSelfAttention, self).__init__()  #调用父类的构造函数
        self.vector_size = vector_size
        self.heads = heads

        self.query = tf.keras.layers.Dense(vector_size//heads)
        self.key = tf.keras.layers.Dense(vector_size//heads)
        self.value = tf.keras.layers.Dense(vector_size//heads)

    def call(self, x):
        batch_size = x.shape[0]#x第0维的数据
        sequence_length = x.shape[1]#x第1维的数据
        heades = self.heads
        vector_size = self.vector_size

        x_mh = tf.reshape(x,[batch_size,sequence_length,heades,vector_size // heades])
        x_mh = tf.transpose(x_mh,[0,2,1,3])
        query = tf.keras.layers.Dense(vector_size // heades)(x_mh)
        key = tf.keras.layers.Dense(vector_size // heades)(x_mh)
        value = tf.keras.layers.Dense(vector_size // heades)(x_mh)
        atten = tf.matmul(query,key,transpose_b=True)
        atten = atten / tf.sqrt(tf.cast(vector_size // heades,dtype=tf.float32))
        atten = tf.nn.softmax(atten)
        atten = tf.matmul(atten,value)
        y = tf.transpose(atten,[0,2,1,3])
        y = tf.reshape(y,[batch_size,sequence_length,vector_size])
        print("-----------call-------------被执行")
        return y

class Transformer(tf.keras.Model):
    def __init__(self,vector_size,heads=1):
        super(Transformer, self).__init__()
        #如结构图定义两个LayerNormalization层
        self.ln0 = tf.keras.layers.LayerNormalization()
        self.ln1 = tf.keras.layers.LayerNormalization()
        #一个多头注意力层
        self.mh_atten = MultiHeadSelfAttention(heads,vector_size)
        #一个Feed Forward层，包含了多个层，所以用Sequential容器
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(vector_size*4,activation=tfa.activations.gelu),#用全连接层将参数变成原先的4倍}
            tf.keras.layers.Dense(vector_size)
        ])

    def call(self,x):
        print("self.mh shape is {}".format(self.mh_atten))
        z = self.ln0(self.mh_atten(x)+x)
        y = self.ln1(z+self.ffn(z))
        return y



batch_size = 4
sequence_length = 10#句子长度
vector_size = 32#每个词的词向量大小
num_heads = 4#将vector_size分为4个部分

x  = tf.random.uniform((batch_size,sequence_length,vector_size))
print(x.shape)
transformer = Transformer(vector_size,num_heads)
y = transformer(x)
print("y[:,0] show :{},y[:,0] shape is :{}".format(y[:,0],y[:,0].shape))

print("\n\n")
print(y.shape)




