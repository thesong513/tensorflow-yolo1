"""
* @program: tensorflow-yolo1
* @description: 
* @author: thesong
* @create: 2020-10-20 14:07
"""

from utils import VOC

if __name__ == '__main__':
    voc = VOC.VOC()
    print(voc.get_voc_size())
    print(voc.toTfrecord())
	