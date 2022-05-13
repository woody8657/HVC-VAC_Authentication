import os
import cv2
import tqdm

if __name__ == '__main__':
    path = '/home/u/woody8657/tmp/val2017/'
    for jpg in tqdm.tqdm(os.listdir(path)):
        img = cv2.imread(os.path.join(path, jpg), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(path, jpg), img)

