"""
* @program: tensorflow2-yolo1
* @description: 
* @author: thesong
* @create: 2020-10-19 10:28
"""

class_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			  "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
class_no = [i for i in range(len(class_name))]

class_dict = dict(zip(class_name, class_no))


num_class = len(class_name)

train_percentage = 0.9

# 数据路径
data_path = "data/VOCdevkit/VOC2012"
image_path = "data/VOCdevkit/VOC2012/JPEGImages"

# 预训练模型路径
pre_model = "model/YOLO_small.ckpt"
# 模型保存路径
moder_save_path = "model"

# tfrecord 数据路径
tfrecord_path = "data/tfrecord/"

train_path = ""
val_path = ""

flipped = True
coord_lambda = 5
no_object_lambda = 0.5

batch_size = 1
epochs = 20

image_size = 448
cell_size = 7
box_per_cell = 2
leaky_relu = 0.1
dropout = 0.5
learning_rate = 0.0001

object_scala = 2.0
no_object_scala = 1.0
class_scala = 2.0
