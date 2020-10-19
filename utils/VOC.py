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


class VOC(ks.utils.Sequence):
	
	def __init__(self):
		self.data_path = setting.data_path
		self.image_size = setting.image_size
		self.cell_size = setting.cell_size
		self.class_name = setting.class_name
		self.class_to_id = setting.class_dict
		self.image_path = setting.image_path
		self.train_percentage = setting.train_percentage
		self.flipped = setting.flipped
		
		# 训练数据label
		self.label_train = None
		# 验证数据label
		self.label_val = None
		self.prepare()
		
		self.image_data_generator = ks.preprocessing.image.ImageDataGenerator()
	
	def prepare(self):
		label_train, label_val = self.load_labels()
		if self.flipped:
			get_label_train_cp = copy.deepcopy(label_train[:len(label_train) // 2])
			for idx in range(len(get_label_train_cp)):
				get_label_train_cp[idx]["flipped"] = True
				get_label_train_cp[idx]["label"] = get_label_train_cp[idx][:, ::-1, :]
				for i in range(self.cell_size):
					for j in range(self.cell_size):
						if get_label_train_cp[idx]["label"][i][j][0] == 1:
							get_label_train_cp[idx]["label"][i][j][1] = self.image_size - 1 \
																		- get_label_train_cp[idx]["label"][i][j][1]
				label_train += get_label_train_cp
			np.random(label_train)
		self.label_train = label_train
		self.label_val = label_val
	
	# 读取 image，并上下翻转
	
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
			imgname = os.path(self.image_path, index + ".jpg")
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
