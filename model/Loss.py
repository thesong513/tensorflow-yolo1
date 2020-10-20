"""
* @program: tensorflow2-yolo1
* @description: 
* @author: thesong
* @create: 2020-10-19 17:38
"""
import numpy as np
import tensorflow as tf
from utils import setting


class Loss():
	
	def __init__(self):
		self.object_class_num = setting.num_class
		self.image_size = setting.image_size
		self.cell_size = setting.cell_size
		self.box_per_cell = setting.box_per_cell
		self.class_num = setting.class_name
		self.coord_lambda = setting.coord_lambda
		self.no_object_lambda = setting.no_object_lambda
	
	def iou(self, box1, box2):
		box1 = tf.stack([box1[:, :, :, :, 0] - box1[:, :, :, :, 2] / 2.0,
						 box1[:, :, :, :, 1] - box1[:, :, :, :, 3] / 2.0,
						 box1[:, :, :, :, 0] + box1[:, :, :, :, 2] / 2.0,
						 box1[:, :, :, :, 1] + box1[:, :, :, :, 3] / 2.0])
		box1 = tf.transpose(box1, [1, 2, 3, 4, 0])
		
		box2 = tf.stack([box2[:, :, :, :, 0] - box2[:, :, :, :, 2] / 2.0,
						 box2[:, :, :, :, 1] - box2[:, :, :, :, 3] / 2.0,
						 box2[:, :, :, :, 0] + box2[:, :, :, :, 2] / 2.0,
						 box2[:, :, :, :, 1] + box2[:, :, :, :, 3] / 2.0])
		box2 = tf.transpose(box2, [1, 2, 3, 4, 0])
		
		left_up = tf.maximum(box1[:, :, :, :, :2], box2[:, :, :, :, :2])
		right_down = tf.minimum(box1[:, :, :, :, 2:], box2[:, :, :, :, 2:])
		
		inter_box = tf.maximum(0.0, right_down - left_up)
		inter_area = inter_box[:, :, :, :, 0] * inter_box[:, :, :, :, 1]
		
		box1_area = (box1[:, :, :, :, 2] - box1[:, :, :, :, 0]) * (box1[:, :, :, :, 3] - box1[:, :, :, :, 1])
		box2_area = (box2[:, :, :, :, 2] - box2[:, :, :, :, 0]) * (box2[:, :, :, :, 3] - box2[:, :, :, :, 1])
		
		union_area = tf.maximum(box1_area + box2_area - inter_area, 1e-10)
		
		return tf.clip_by_value(inter_area / union_area, 0.0, 1.0)
	
	def box_loss(self, true_box, pred_box, object_mask):
		mask = tf.expand_dims(object_mask, 4)
		box_diff = mask * (true_box - pred_box)
		box_loss_mean = self.coord_lambda * tf.reduce_mean(tf.reduce_sum(tf.square(box_diff), axis=[1, 2, 3, 4]),
														   name='box_loss')
		
		return box_loss_mean
	
	def confidence_loss(self, pred_confidence, object_mask):
		no_object_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
		
		object_diff = object_mask * (1.0 - pred_confidence)
		object_loss_mean = tf.reduce_mean(tf.reduce_sum(tf.square(object_diff), axis=[1, 2, 3]), name='object_loss')
		
		no_object_diff = no_object_mask * pred_confidence
		no_object_loss_mean = self.no_object_lambda * tf.reduce_mean(
			tf.reduce_sum(tf.square(no_object_diff), axis=[1, 2, 3]), name='no_object_loss')
		
		return object_loss_mean, no_object_loss_mean
	
	def class_loss(self, true_class, pred_class, true_object):
		class_diff = true_object * (true_class - pred_class)
		class_loss_mean = tf.reduce_mean(tf.reduce_sum(tf.square(class_diff), axis=[1, 2, 3]), name='class_loss')
		
		return class_loss_mean
	
	def loss(self, y_true, y_pred):
		index_confidence = self.box_per_cell
		index_class = self.box_per_cell * (4 + 1)
		
		pred_confidence = tf.reshape(y_pred[:, :, :, :index_confidence],
									 [-1, self.cell_size, self.cell_size, self.box_per_cell])
		pred_box = tf.reshape(y_pred[:, :, :, index_confidence:index_class],
							  [-1, self.cell_size, self.cell_size, self.box_per_cell, 4])
		pred_class = tf.reshape(y_pred[:, :, :, index_class:],
								[-1, self.cell_size, self.cell_size, self.object_class_num])
		
		true_object = tf.reshape(y_true[:, :, :, 0], [-1, self.cell_size, self.cell_size, 1])
		true_box_image = tf.reshape(y_true[:, :, :, 1:5], [-1, self.cell_size, self.cell_size, 1, 4])
		true_box_image = tf.tile(true_box_image, [1, 1, 1, self.box_per_cell, 1])
		true_class = y_true[:, :, :, 5:]
		
		offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.box_per_cell),
										 (self.box_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
		offset = tf.constant(offset, dtype=tf.float32)
		offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.box_per_cell])
		
		pred_box_grid = tf.stack([pred_box[:, :, :, :, 0],
								  pred_box[:, :, :, :, 1],
								  tf.sqrt(pred_box[:, :, :, :, 2]),
								  tf.sqrt(pred_box[:, :, :, :, 3])])
		pred_box_grid = tf.transpose(pred_box_grid, [1, 2, 3, 4, 0])
		
		pred_box_image = tf.stack([(pred_box_grid[:, :, :, :, 0] + offset) / self.cell_size,
								   (pred_box_grid[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
								   tf.square(pred_box_grid[:, :, :, :, 2]),
								   tf.square(pred_box_grid[:, :, :, :, 3])])
		pred_box_image = tf.transpose(pred_box_image, [1, 2, 3, 4, 0])
		
		true_box_grid = tf.stack([true_box_image[:, :, :, :, 0] * self.cell_size - offset,
								  true_box_image[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
								  tf.sqrt(true_box_image[:, :, :, :, 2]),
								  tf.sqrt(true_box_image[:, :, :, :, 3])])
		true_box_grid = tf.transpose(true_box_grid, [1, 2, 3, 4, 0])
		
		true_pred_iou = self.iou(true_box_image, pred_box_image)
		
		object_mask = tf.reduce_max(true_pred_iou, 3, keepdims=True)
		object_mask = true_object * tf.cast((true_pred_iou >= object_mask), dtype=tf.float32)
		
		box_loss_mean = self.box_loss(true_box_grid, pred_box_grid, object_mask)
		object_loss_mean, no_object_loss_mean = self.confidence_loss(pred_confidence, object_mask)
		class_loss_mean = self.class_loss(true_class, pred_class, true_object)
		return box_loss_mean + object_loss_mean + no_object_loss_mean + class_loss_mean
