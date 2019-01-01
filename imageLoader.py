import cv2
import numpy as np
import os
from imageProcessor import SimplePreprocessor

def image2vector (filename):
    returnVect=zeros((1,1024))
    f=open(filename)
    for i in range (32):
       lineStr =fr.readline()
       for j in range (32):
            returnVect[0,32*i*j]=int(lineStr[j])
    return returnVect

class SimpleDatasetLoader:
    def __init__(self,preprocessors = None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []


    def loader(self,imgPaths,verbose = -1):
        #initial the features and labels
        data = []
        labels = []

        #loop over the input images
        for (i,imgPath) in enumerate(imgPaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            img = cv2.imread(imgPath)
            # label = imgPath.split(os.path.sep)[-2]
            label = imgPath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    img = p.preprocessors(img)
            #这里将img（32，32,3） reshape成(1,3072)
            data.append(np.array(img).reshape((1,3072)))
            labels.append(label)
            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,len(imgPaths)))
        #循环结束：data.shape（4000,1,3072）
        return (np.array(data),np.array(labels))