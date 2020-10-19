"""
* @program: tensorflow2-yolo1
* @description: 
* @author: thesong
* @create: 2020-10-19 14:36
"""

import tensorflow as tf
import numpy as np
from utils import VOC, setting

tfrecord_path = setting.tfrecord_path

voc = VOC.VOC()
writer = tf.python_io.TFRecordWriter(tfrecord_path + "train.tfrecords")

for i in range(voc.label_train):
	imgname = voc.label_train[i]["iname"]
	flipped = voc.label_train[i]["flipped"]
	image = voc.read_image(imgname, flipped)
	label = voc.label_train[i]["label"]
	
	label_raw = label.tobytes()
	bytes_list_label = tf.train.BytesList(value=[label_raw])
	label_feature = tf.train.Feature(bytes_list=bytes_list_label)
	image_raw = image.tobytes()
	bytes_list_image = tf.train.BytesList(value=[image_raw])
	image_feature = tf.train.Feature(bytes_lits=bytes_list_image)
	feature = tf.train.Features(feature={
		"label": label_feature,
		"img_raw": image_feature
	})
	example = tf.train.Example(feature)
	writer.write(example.SerializeToString())

writer.close()

writer = tf.python_io.TFRecordWriter(tfrecord_path + "val.tfrecords")

for k in range(len(voc.label_val)):
	imname = voc.label_val[k]['imname']
	flipped = voc.label_val[k]['flipped']
	image = voc.read_image(imname, flipped)
	label = voc.label_val[k]['label']
	
	label_raw = label.tobytes()
	img_raw = image.tobytes()  # 将图片转化为原生bytes
	example = tf.train.Example(features=tf.train.Features(feature={
		"label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
		'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
	}))
	writer.write(example.SerializeToString())  # 序列化为字符串
	print(k)

writer.close()
