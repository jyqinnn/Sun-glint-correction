import sys, os, argparse
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import torch
import numpy as np
import cv2


def FMM(img, mask):
    
    imgg = np.array(img,dtype=np.uint8)
    maskk = np.array(mask,dtype=np.uint8)
    img_res = cv2.inpaint(imgg,maskk,10,cv2.INPAINT_TELEA)
    
    return img_res
      
        
def parse_arges():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_img', type=str,
                        default='./data/glint/0.png')
    parser.add_argument('--test_mask', type=str,
                        default='./data/label/0.png')
    parser.add_argument('--output_path', type=str,
                        default='./result/frame_inpaint/glint/0.png')

    args = parser.parse_args()
    return args


def main():

    args = parse_arges()

    image = cv2.imread(args.test_img)
    mask = cv2.imread(args.test_mask, cv2.IMREAD_UNCHANGED)
    
    img_res = FMM(image, mask)

    cv2.imwrite(args.output_path, img_res)
    print('Result Saved')

if __name__ == '__main__':
    main()
