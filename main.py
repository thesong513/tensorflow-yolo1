"""
* @program: tensorflow-yolo1
* @description: 
* @author: thesong
* @create: 2020-10-20 16:42
"""
from utils import VOC
from model import Model
import tensorflow as tf


def main():
	# voc = VOC.VOC()
	# train_tfrecords, val_tfrecords = voc.toTfrecord()
	# train_size, val_size = voc.get_voc_size()
	# print(train_tfrecords, val_tfrecords, train_size, val_size)
	# data/tfrecord/train.tfrecords data/tfrecord/val.tfrecords 23118 1713
	with tf.device("/cpu:0"):
		model = Model.Model()
		model.summary()
		
		# history = model.train(train_tfrecords, val_tfrecords, train_size, val_size)
		history = model.train("data/tfrecord/train.tfrecords", "data/tfrecord/val.tfrecords", 23118, 1713)

if __name__ == '__main__':
	main()
