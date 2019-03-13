# coding: utf-8

# In[1]:

# Author: Li
import tensorflow as tf
import os.path
import argparse
from tensorflow.python.framework import graph_util
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
windonws = 0
if windonws:
    MODEL_DIR = "F:\\blf_facenet\\pb_file\\"
    MODEL_NAME = "vgg_face_need_white_201805230.pb"
    pb_path = "F:\\blf_facenet\\pb_file\\" + MODEL_NAME
    model_save_path = "F:\\blf_facenet\\cnn_model2\\"  # 网络模型文件 保存路径
else:
    MODEL_DIR = './model'
    MODEL_NAME = 'detector.pb'
    pb_path = os.path.join(MODEL_DIR , MODEL_NAME)
    model_save_path = './model'

if not tf.gfile.Exists(MODEL_DIR):  # 创建目录
    tf.gfile.MakeDirs(MODEL_DIR)

output_node_names = "lztower_0/images,lztower_0/output/boxes,lztower_0/output/scores,lztower_0/output/labels,training_flag"  # 原模型输出操作节点的名字


# In[2]:

def freeze_graph(model_folder):
    print("start")
    checkpoint = tf.train.get_checkpoint_state(model_folder)  # 检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path  # 得ckpt文件路径
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME)  # PB模型保存路径

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)  # 得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.

    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据

        #         print("predictions : ", sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]})) # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字
        # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

        [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",")  # 如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
    print("~~~~")


#         print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

#         for op in graph.get_operations():
#             print(op.name, op.values())


# In[4]:

if __name__ == '__main__':
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument("model_folder", type=str, help="input ckpt model dir") #命令行解析，help是提示符，type是输入的类型，
    # 这里运行程序时需要带上模型ckpt的路径，不然会报 error: too few arguments
    #     aggs = parser.parse_args()
    freeze_graph(model_save_path)
    # freeze_graph("model/ckpt") #模型目录

    print("Well done!")

# In[5]:

import cv2
import numpy as np

image_size = 112  # 数据集样本的 size:24*24
image_heatmap_size = 32  # 数据集样本的 size:24*24
input_img_channel = 3  # 输入图像通道数

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def feature_distance(features_left, features_right):
    if len(features_left) == 0 or len(features_right) == 0:
        return np.empty((0))
    return np.linalg.norm(features_left - features_right, axis=1)


def recognize(a, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        # print("打开.pb 文件")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # 获取训练好的模型 的 tensor和op
            x = sess.graph.get_tensor_by_name('images:0')

            keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
            embeddings = sess.graph.get_tensor_by_name('embeddings:0')

            img = cv2.imread(a)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            #
            std_img = prewhiten(img)
            std_img = std_img.reshape(1, image_size, image_size, input_img_channel)
            blf_embeddings1 = sess.run(embeddings, feed_dict={x: std_img, keep_prob: 1.0})  # 进行模型迭代训练
            print (blf_embeddings1)


print("function initial~~~")

if windonws:
    root_image_path = "C:\\Users\\koke8\\Desktop\\20180529\\test\\"
else:
    root_image_path = 'data/log.jpg'
#recognize(root_image_path, pb_path)

# cv2.destroyAllWindows()#销毁opencv显示窗口
print("well done")
# f.close()


# In[ ]:
