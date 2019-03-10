import cv2
import os
import time
import numpy as np
from api.frcnn_detector import FrcnnDetector
from tools.to_lableimg import to_xml
from tools.to_labelme import to_json
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
detector = FrcnnDetector('./model/detector.pb')

import os
def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            # if s == "pts":
            #     continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList


def faceboxes_with_landmark():
    count = 0
    data_dir = '/home/lz/FACE/widerface/img'

    pics = []
    GetFileList(data_dir,pics)

    pics = [x for x in pics if 'jpg' in x or 'png' in x]
    #pics.sort()

    for pic in pics:

        img=cv2.imread(pic)

        img_show = img.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        star=time.time()
        boxes,scores=detector(img,0.0)
        #print('one iamge cost %f s'%(time.time()-star))
        #print(boxes.shape)
        #print(boxes)  
        ################toxml or json



        for box_index in range(boxes.shape[0]):

            bbox = boxes[box_index]

            cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 0, 0), 4)

        cv2.namedWindow('res',0)
        cv2.imshow('res',img_show)
        cv2.waitKey(0)
    print(count)


if __name__=='__main__':
    faceboxes_with_landmark()
