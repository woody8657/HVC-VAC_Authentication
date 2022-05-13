import cv2
import numpy as np
from PIL import Image
from utils import decode_shares

def make_gif(share1, share2):
    share = [share1, share2]
    shape = (share1.shape[0],share1.shape[1]*2)
    tmp = np.zeros(shape)
    gif = []
    for i in range(share1.shape[1]+1):

        try:
            tmp = np.ones(shape)*255
            tmp[:,:np.concatenate((share[0][:,:share1.shape[1]-i],decode_shares(share[0][:,-i:], share[1][:,:i]),share[1][:,i:]), axis=1).shape[1]] = np.concatenate((share[0][:,:share1.shape[1]-i],decode_shares(share[0][:,-i:], share[1][:,:i]),share[1][:,i:]), axis=1)
    
            gif.append(tmp)
        
        except:
            try:
                tmp = np.ones(shape)*255
                tmp[:, :np.concatenate((share[0][:,:share1.shape[1]-i],share[1][:,i:]), axis=1).shape[1]] = np.concatenate((share[0][:,:share1.shape[1]-i],share[1][:,i:]), axis=1)
            
                gif.append(tmp)

            except:
                tmp = np.ones(shape)*255
                tmp[:,:share1(share[0][:,-i:], share[1][:,:i]).shape[1]] = decode_shares(share[0][:,-i:], share[1][:,:i])
            
                gif.append(tmp)
        

    gif = [np.expand_dims(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2RGB),axis=0) for frame in gif]
    gif = gif + [gif[-1] for _ in range(300)]
    gif = np.concatenate(gif)
    print(gif.shape)
    gif = [Image.fromarray(frame) for frame in gif]
    gif[0].save("./figures/demo.gif", save_all=True, append_images=gif[1:], duration=20, loop=3)

if __name__ == '__main__':
    share1 = cv2.imread('/home/u/woody8657/projs/HVC-VAC_Authentication/coco_test/outputs/share1.png', cv2.IMREAD_GRAYSCALE)
    share2 = cv2.imread('/home/u/woody8657/projs/HVC-VAC_Authentication/coco_test/outputs/share20.png', cv2.IMREAD_GRAYSCALE)

    make_gif(share1, share2)