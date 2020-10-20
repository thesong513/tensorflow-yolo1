"""
* @program: tensorflow2-yolo1
* @description: 
* @author: thesong
* @create: 2020-10-19 10:28
"""
import copy

from utils import setting
import cv2
import os
import random
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow.keras as ks
import tensorflow as tf


class VOC():
	
	def __init__(self):
		self.data_path = setting.data_path
		self.image_size = setting.image_size
		self.cell_size = setting.cell_size
		self.class_name = setting.class_name
		self.class_to_id = setting.class_dict
		self.image_path = setting.image_path
		self.train_percentage = setting.train_percentage
		self.flipped = setting.flipped
		self.tfrecord_path = setting.tfrecord_path
		
		# 训练数据label
		self.label_train = None
		# 验证数据label
		self.label_val = None
		self.prepare()
		
	
	def prepare(self):
		label_train, label_val = self.load_labels()
		if self.flipped:
			label_train_cp = copy.deepcopy(label_train[:len(label_train) // 2])
			# 取一半的label_train[{imagename,label,flipped}]
			for idx in range(len(label_train_cp)):
				# 遍历这一半的 label_train
				label_train_cp[idx]["flipped"] = True
				label_train_cp[idx]["label"] = label_train_cp[idx][:, ::-1, :]
				for i in range(self.cell_size):
					for j in range(self.cell_size):
						if label_train_cp[idx]["label"][i][j][0] == 1:
							# box 中心点 x 的坐标
							label_train_cp[idx]["label"][i][j][1] = self.image_size - 1 \
																	- label_train_cp[idx]["label"][i][j][1]
				label_train += label_train_cp
			np.random.shuffle(label_train)
		self.label_train = label_train
		self.label_val = label_val
	
	# 获取训练集总数和验证集总数
	def get_voc_size(self):
		train_size = len(self.label_train)
		val_size = len(self.label_val)
		return train_size, val_size
	
	# 读取 image，并左右翻转
	def read_image(self, imgname, flipped=False):
		image = cv2.imread(imgname)
		image = cv2.resize(image, (self.image_size, self.image_size))
		
		if flipped:
			image = image[:, ::-1, :]
		return image
	
	def load_labels(self):
		image_index = os.listdir(self.image_path)
		image_index = [i.replace(".jpg", "") for i in image_index]
		random.shuffle(image_index)
		
		# 划分 训练集 验证集
		train_index = int(len(image_index) * self.train_percentage)
		image_index_train = image_index[:train_index]
		image_index_val = image_index[train_index:]
		labels_train = []
		labels_val = []
		
		for index in image_index_train:
			label, num = self.load_pascal_annotation(index)
			if num == 0:
				continue
			imgname = os.path.join(self.image_path, index + ".jpg")
			labels_train.append({"imgname": imgname, "label": label, "flipped": False})
		
		for index in image_index_val:
			label, num = self.load_pascal_annotation(index)
			if num == 0:
				continue
			imgname = os.path.join(self.image_path, index + ".jpg")
			labels_val.append({"imgname": imgname, "label": label, "flipped": False})
		
		return labels_train, labels_val
	
	def load_pascal_annotation(self, index):
		imgname = os.path.join(self.image_path, index + ".jpg")
		img = cv2.imread(imgname)
		
		h_ratio = 1.0 * self.image_size / img.shape[0]
		w_ratio = 1.0 * self.image_size / img.shape[1]
		
		label = np.zeros((self.cell_size, self.cell_size, 25))
		filename = os.path.join(self.data_path, "Annotations", index + ".xml")
		et = ET.parse(filename)
		objs = et.findall("object")
		
		for obj in objs:
			bbox = obj.find("bndbox")
			x1 = max(min((float(bbox.find("xmin").text) - 1) * w_ratio, self.image_size - 1), 0)
			y1 = max(min((float(bbox.find("ymin").text) - 1) * h_ratio, self.image_size - 1), 0)
			x2 = max(min((float(bbox.find("xmax").text) - 1) * w_ratio, self.image_size - 1), 0)
			y2 = max(min((float(bbox.find("ymax").text) - 1) * w_ratio, self.image_size - 1), 0)
			
			class_id = self.class_to_id[obj.find("name").text.lower().strip()]
			
			box = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]
			
			x_id = int(box[0] * self.cell_size / self.image_size)
			y_id = int(box[1] * self.cell_size / self.image_size)
			if label[y_id, x_id, 0] == 1:
				continue
			label[y_id, x_id, 0] = 1
			label[y_id, x_id, 1:5] = box
			label[y_id, x_id, 5 + class_id] = 1
		
		return label, len(objs)
	
	# 转化为 tfrecord
	def toTfrecord(self):
		train_writer = tf.python_io.TFRecordWriter(self.tfrecord_path + "train.tfrecords")
		for i in range(self.label_train):
			imgname = self.label_train[i]["iname"]
			flipped = self.label_train[i]["flipped"]
			image = self.read_image(imgname, flipped)
			label = self.label_train[i]["label"]
			
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
			train_writer.write(example.SerializeToString())
		
		train_writer.close()
		
		val_writer = tf.python_io.TFRecordWriter(self.tfrecord_path + "val.tfrecords")
		
		for k in range(len(self.label_val)):
			imname = self.label_val[k]['imname']
			flipped = self.label_val[k]['flipped']
			image = self.read_image(imname, flipped)
			label = self.label_val[k]['label']
			
			label_raw = label.tobytes()
			img_raw = image.tobytes()  # 将图片转化为原生bytes
			example = tf.train.Example(features=tf.train.Features(feature={
				"label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
				"img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
			}))
			val_writer.write(example.SerializeToString())  # 序列化为字符串
		
		val_writer.close()
		return self.tfrecord_path + "train.tfrecords", self.tfrecord_path + "val.tfrecords"
