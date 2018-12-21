#coding:utf-8
# 导入工具包
import numpy as np
import tensorflow as tf
import os
import csv

# 读入训练数据
print("开始导入训练数据 apple_sugar_train_data.csv")

# OSS存放文件的目录，需根据具体情况进行调整
file_dir = "oss://bucket-star-958hd2.oss-cn-shanghai.aliyuncs.com/SL3303/u-fxfv7s29/"

# 捕获输入文件的整个名称（含目录）
dirname = os.path.join(file_dir,"apple_sugar_train_data.csv")

filename_queue = tf.train.string_input_producer([dirname])

#定义reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 定义矩阵格式以及数据类型，list形式
record_defaults = [[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]    ,[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]     ,[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]     ,[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]     ,[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]  ]
# 解析出的每一个属性都是rank为0的标量
col1, col2, col3, col4, col5 ,col6, col7, col8, col9, col10, col11, col12, col13, col14, col15 ,col16, col17, col18, col19, col20, col21, col22, col23, col24, col25 ,col26, col27, col28, col29, col30, col31, col32, col33, col34, col35 ,col36, col37, col38, col39, col40, col41, col42, col43, col44, col45 ,col46, col47, col48, col49, col50, col51, col52, col53, col54, col55 ,col56, col57, col58, col59, col60,  col61, col62, col63, col64, col65 ,col66, col67, col68, col69, col70, col71, col72, col73, col74, col75 ,col76, col77, col78, col79, col80, col81, col82, col83, col84, col85 ,col86, col87, col88, col89, col90  = tf.decode_csv(value, record_defaults=record_defaults)
# 将88个光谱信息能力值进行拼接
features = tf.stack([col3, col4, col5 ,col6, col7, col8, col9, col10, col11, col12, col13, col14, col15 ,col16, col17, col18, col19, col20, col21, col22, col23, col24, col25 ,col26, col27, col28, col29, col30, col31, col32, col33, col34, col35 ,col36, col37, col38, col39, col40, col41, col42, col43, col44, col45 ,col46, col47, col48, col49, col50, col51, col52, col53, col54, col55 ,col56, col57, col58, col59, col60,  col61, col62, col63, col64, col65 ,col66, col67, col68, col69, col70, col71, col72, col73, col74, col75 ,col76, col77, col78, col79, col80, col81, col82, col83, col84, col85 ,col86, col87, col88, col89, col90])
features_target = tf.stack([col2])

# 初始化数组
x_vals_train = np.zeros( (67,88) )
y_vals_train = np.zeros( (67,) )

with tf.Session() as sess:
		#线程协调器
    coord = tf.train.Coordinator()
    #启动线程
    threads = tf.train.start_queue_runners(coord=coord)
		#将训练数据的x部分列表矩阵化
    for i in range(67):
        example  = sess.run([features])
        example_mat = np.mat(example[0])
        x_vals_train[i] =  example_mat
		
		#将训练数据的y部分列表矩阵化
    for j in range(67):
        example_target  = sess.run([features_target])
        example_target_mat = np.mat(example_target[0][0])
        y_vals_train[j] =  example_target_mat

		#循环结束后，请求关闭所有线程
    coord.request_stop()
    coord.join(threads)
    sess.close()

print ('训练数据如下：')
print (x_vals_train)
print (y_vals_train)


# 读入测试数据
print("开始导入测试数据 apple_sugar_check_data.csv")

# 捕获输入文件的整个名称（含目录）
dirname2 = os.path.join(file_dir, "apple_sugar_check_data.csv")
filename_queue2 = tf.train.string_input_producer([dirname2])

#定义reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue2)

# 定义矩阵格式以及数据类型，list形式
record_defaults = [[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]    ,[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]     ,[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]     ,[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]     ,[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]  ]
# 解析出的每一个属性都是rank为0的标量
col1, col2, col3, col4, col5 ,col6, col7, col8, col9, col10, col11, col12, col13, col14, col15 ,col16, col17, col18, col19, col20, col21, col22, col23, col24, col25 ,col26, col27, col28, col29, col30, col31, col32, col33, col34, col35 ,col36, col37, col38, col39, col40, col41, col42, col43, col44, col45 ,col46, col47, col48, col49, col50, col51, col52, col53, col54, col55 ,col56, col57, col58, col59, col60,  col61, col62, col63, col64, col65 ,col66, col67, col68, col69, col70, col71, col72, col73, col74, col75 ,col76, col77, col78, col79, col80, col81, col82, col83, col84, col85 ,col86, col87, col88, col89, col90  = tf.decode_csv(value, record_defaults=record_defaults)
# 将88个光谱信息能力值进行拼接
features2 = tf.stack([col3, col4, col5 ,col6, col7, col8, col9, col10, col11, col12, col13, col14, col15 ,col16, col17, col18, col19, col20, col21, col22, col23, col24, col25 ,col26, col27, col28, col29, col30, col31, col32, col33, col34, col35 ,col36, col37, col38, col39, col40, col41, col42, col43, col44, col45 ,col46, col47, col48, col49, col50, col51, col52, col53, col54, col55 ,col56, col57, col58, col59, col60,  col61, col62, col63, col64, col65 ,col66, col67, col68, col69, col70, col71, col72, col73, col74, col75 ,col76, col77, col78, col79, col80, col81, col82, col83, col84, col85 ,col86, col87, col88, col89, col90])
features_target2 = tf.stack([col2])

# 初始化数组
x_vals_test = np.zeros( (26,88) )
y_vals_test = np.zeros( (26,) )

with tf.Session() as sess:
		#线程协调器
    coord = tf.train.Coordinator()
    #启动线程
    threads = tf.train.start_queue_runners(coord=coord)
		#将测试数据的x部分列表矩阵化
    for k in range(26):
        example2  = sess.run([features2])
        example_mat2 = np.mat(example2[0])
        x_vals_test[k] =  example_mat2
		
		#将测试数据的y部分列表矩阵化
    for l in range(26):
        example_target2  = sess.run([features_target2])
        example_target2_mat = np.mat(example_target2[0][0])
        y_vals_test[l] =  example_target2_mat

		#循环结束后，请求关闭所有线程
    coord.request_stop()
    coord.join(threads)
    sess.close()

print ('测试数据如下：')
print (x_vals_test)
print (y_vals_test)

# 捕获训练x集、测试x集的均值及标准差
x_mean = np.concatenate([x_vals_train, x_vals_test]).mean(axis=0)
x_std = np.concatenate([x_vals_train, x_vals_test]).std(axis=0)

# 得到训练x集、测试x集的z-score值
x_vals_train = (x_vals_train - x_mean) / x_std
x_vals_test = (x_vals_test - x_mean) / x_std

# 导入工具包
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 创建会话
sess = tf.Session()

# 设置种子
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)

# 创建一个神经网络
def _dense( x, outsize, activation=None, 
    kernel_initializer=tf.contrib.layers.xavier_initializer(),
    name=None ):
    return tf.layers.dense( x, outsize, 
        activation=activation, 
        kernel_initializer=kernel_initializer,
        name=name )

# 设置数据形状及数据类型
x_data = tf.placeholder(shape=[None, 88], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 激活函数
net = _dense( x_data, 512, activation=tf.nn.relu )
net = _dense( net, 128, activation=tf.nn.relu )
final_output = _dense( net, 1 )

# 损失函数定义为均方差
loss = tf.reduce_mean(tf.square(y_target - final_output))

# 声明梯度下降算法，学习率为0.0004，最大限度地减少损失
my_opt = tf.train.GradientDescentOptimizer(0.0004)
train_step = my_opt.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

# 初始化两个列表（list）存储训练损失和测试损失
loss_vec = []
test_loss = []

# 设置batch_size大小
batch_size=67

# 迭代5000次训练神经网络，此次实验在迭代的过程中发现5000左右基本可以是测试集的均方差达到比较低的值，继续迭代的话，训练集的均方差会继续降低，但是测试集均方差无明显降低
for i in range(5000):
    _, temp_loss = sess.run([train_step, loss], feed_dict={x_data: x_vals_train, y_target: y_vals_train.reshape((-1, 1))})

		# 追加训练损失值
    loss_vec.append(np.sqrt(temp_loss))
    
    # 本次训练对应的测试集损失值
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    # 追加测试损失值
    test_loss.append(np.sqrt(test_temp_loss))
    
    #每200次迭代，输出一行信息
    if (i+1)%200==0:
        print('Generation: {:>5} . Train_Loss: {:>.4f}, Test_Loss: {:>.4f} '.format( i+1, temp_loss, test_temp_loss) )

# 基于最后一次的神经网络模型，输出测试集对应的预测结果
output = sess.run( final_output, feed_dict={x_data: x_vals_test} )

# 打印信息
print('MSE:',test_temp_loss)
print('预测数据如下:')
print(output)


# 输出数据对应的OSS文件名称
dirname_out = os.path.join(file_dir, "apple_sugar_forecast_data.txt")

# 处理输出数据
file_out = [['MSE',test_temp_loss]]
for i in range(26):
    x = [i+1,output[i][0]]
    file_out.append(x)

file_out_2 = np.array(file_out)

# 输出数据至目标文件
tf.gfile.FastGFile(dirname_out, 'wb').write(format(file_out_2))
