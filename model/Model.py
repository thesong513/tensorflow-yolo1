"""
* @program: tensorflow2-yolo1
* @description: 
* @author: thesong
* @create: 2020-10-19 15:09
"""

import tensorflow as tf
from utils import setting
import tensorflow.keras as ks
from tensorflow.keras import layers
from model import Loss


class Model():
	def __init__(self):
		self.image_size = setting.image_size
		self.leaky_relu = setting.leaky_relu
		self.dropout = setting.dropout
		self.cell_size = setting.cell_size
		self.box_per_cell = setting.box_per_cell
		self.class_num = setting.class_name
		self.learning_rate = setting.learning_rate
		self.moder_save_path = setting.moder_save_path
		self.epochs = setting.epochs
		self.model = self._build_model()
		self.optimizer = self._optimizer()
		self.loss = self._loss()
	
	def _build_model(self):
		model = ks.Sequential()
		
		model.add(layers.InputLayer(input_shape=(self.image_size, self.image_size, 3), name="input"))
		
		model.add(
			layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", use_bias=False, name="conv1"))
		model.add(layers.BatchNormalization(name="norm1"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, name="pool1"))
		
		model.add(layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv2"))
		model.add(layers.BatchNormalization(name="norm2"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, name="pool2"))
		
		model.add(layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False,
								name="conv3"))
		model.add(layers.BatchNormalization(name="norm3"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv4"))
		model.add(layers.BatchNormalization(name="norm4"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False,
								name="conv5"))
		model.add(layers.BatchNormalization(name="norm5"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv6"))
		model.add(layers.BatchNormalization(name="norm6"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False,
								name="conv7"))
		model.add(layers.BatchNormalization(name="norm7"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv8"))
		model.add(layers.BatchNormalization(name="norm8"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False,
								name="conv9"))
		model.add(layers.BatchNormalization(name="norm9"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv10"))
		model.add(layers.BatchNormalization(name="norm10"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False,
								name="conv11"))
		model.add(layers.BatchNormalization(name="norm11"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv12"))
		model.add(layers.BatchNormalization(name="norm12"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False,
								name="conv13"))
		model.add(layers.BatchNormalization(name="norm13"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv14"))
		model.add(layers.BatchNormalization(name="norm14"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False,
								name="conv15"))
		model.add(layers.BatchNormalization(name="norm15"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv16"))
		model.add(layers.BatchNormalization(name="norm16"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, name="pool16"))
		
		model.add(layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False,
								name="conv17"))
		model.add(layers.BatchNormalization(name="norm17"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv18"))
		model.add(layers.BatchNormalization(name="norm18"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False,
								name="conv19"))
		model.add(layers.BatchNormalization(name="norm19"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv20"))
		model.add(layers.BatchNormalization(name="norm20"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv21"))
		model.add(layers.BatchNormalization(name="norm21"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv22"))
		model.add(layers.BatchNormalization(name="norm22"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv23"))
		model.add(layers.BatchNormalization(name="norm23"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
								name="conv24"))
		model.add(layers.BatchNormalization(name="norm24"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		
		model.add(layers.Flatten())
		model.add(layers.Dense(units=4096, name="fc26"))
		model.add(layers.LeakyReLU(self.leaky_relu))
		model.add(layers.Dropout(self.dropout))
		
		model.add(layers.Dense(units=(self.cell_size * self.cell_size * (self.box_per_cell * (4 + 1) + self.num_class)),
							   name="fc27"))
		model.add(layers.Softmax(name="softmax27"))
		
		model.add(layers.Reshape((self.cell_size, self.cell_size, (self.box_per_cell * (4 + 1) + self.num_class)),
								 name="reshape27"))
		
		return model
	
	def _loss(self):
		loss = Loss.Loss().loss
		return loss
	
	def _optimizer(self):
		optimizer = ks.optimizers.Adam(lr=self.learning_rate)
		return optimizer
	
	def _compile(self):
		self.model.compile(optimizer=self.optimizer, loss=self.loss)
	
	def train(self):
		train_generator = dg.DataGenerator(train_data_path, cfg.batch_size, train_transform_param)
		validation_generator = dg.DataGenerator(validation_data_path, cfg.batch_size)
		
		earlystop = ks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='min')
		reducelr = ks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto',
												  min_delta=0.0001, cooldown=0, min_lr=0)
		tensorboard = ks.callbacks.TensorBoard(log_dir='log', histogram_freq=0, batch_size=cfg.batch_size,
											   write_graph=True, write_images=False)
		self.model.fit_generator(
			epochs=self.epochs,
			generator=train_generator,
			validation_data=validation_generator,
			callbacks=[earlystop, reducelr, tensorboard]
		)
		self.model.save_weights(self.moder_save_path)
	
	def summary(self):
		self.model.summary()
