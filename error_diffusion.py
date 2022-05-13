import os
import cv2
from cv2 import IMREAD_GRAYSCALE
import numpy as np

def error_diffusion(img):
    E = np.zeros(img.shape)
    G = np.zeros(img.shape)
    for i in range(0,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            G[i,j] = ((img[i,j]+E[i,j]) >= 255/2) * 255
            tmp = (img[i,j]+E[i,j]) - G[i,j]
            E[i:i+2,j-1:j+2] = E[i:i+2,j-1:j+2] + np.array([[0,0,7],[3,5,1]])/16*tmp
    return G / 255

if __name__ == '__main__':
    img_path = '/home/u/woody8657/projs/HVC-VAC_Authentication/coco_test/inputs/tmp/000000000724.jpg'
    img = cv2.imread(img_path, IMREAD_GRAYSCALE)
    img = error_diffusion(img)
    cv2.imwrite(img_path, img)
    