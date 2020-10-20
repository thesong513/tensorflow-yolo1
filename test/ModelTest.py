"""
* @program: tensorflow-yolo1
* @description: 
* @author: thesong
* @create: 2020-10-20 09:38
"""

from model import Model
import tensorflow as tf

if __name__ == '__main__':
    with tf.device("/cpu:0"):
        yolo = Model.Model()
        yolo.summary()
        