import os
from pathlib import Path
import glob
import cv2
import numpy as np
from tqdm import tqdm

DATASET_PATH = "./val2017"
PREPROCESS_DATASET_PATH = "./val2017_preprocess"
IMG_HEIGH = 640
IMG_WIDTH = 640
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32, return_int=False):
    '''Resize and pad image while meeting stride-multiple constraints.'''
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
       new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if not return_int:
        return im, r, (dw, dh)
    else:
        return im, r, (left, top)

def precess_image(img_src, img_size):
    """
        Process image before image inference.
    """
    image = letterbox(img_src, img_size, auto=False)[0]
    return image

class LoadData:
    def __init__(self, path, webcam, webcam_addr):
        self.webcam = webcam
        self.webcam_addr = webcam_addr
        if webcam: # if use web camera
            imgp = []
            vidp = [int(webcam_addr) if webcam_addr.isdigit() else webcam_addr]
        else:
            p = str(Path(path).resolve())  # os-agnostic absolute path
            if os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, '**/*.*'), recursive=True))  # dir
            elif os.path.isfile(p) and p[-3:] != 'txt':
                files = [p]  # files
            elif os.path.isfile(p) and p[-3:] == 'txt':
                files = open(p, 'r').readlines()
                files = [x.strip() for x in files]
                print(f'Found {len(files)} images in {p}')
            else:
                raise FileNotFoundError(f'Invalid path {p}')
            imgp = [i for i in files if i.split('.')[-1] in IMG_FORMATS]
            vidp = [v for v in files if v.split('.')[-1] in VID_FORMATS]
        self.files = imgp + vidp
        self.nf = len(self.files)
        self.type = 'image'
        if len(vidp) > 0:
            self.add_video(vidp[0])  # new video
        else:
            self.cap = None

    # @staticmethod
    def checkext(self, path):
        if self.webcam:
            file_type = 'video'
        else:
            file_type = 'image' if path.split('.')[-1].lower() in IMG_FORMATS else 'video'
        return file_type

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.checkext(path) == 'video':
            self.type = 'video'
            ret_val, img = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.add_video(path)
                ret_val, img = self.cap.read()
        else:
            # Read image
            self.count += 1
            img = cv2.imread(path)  # BGR
        return img, path, self.cap
        
    def add_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def __len__(self):
        return self.nf  # number of files


if __name__ == "__main__":
    dataset = LoadData(DATASET_PATH, False, 0)
    os.makedirs(PREPROCESS_DATASET_PATH, exist_ok=True)
    for img, path, cap in tqdm(dataset):
        preprocess_img = precess_image(img, (IMG_HEIGH, IMG_WIDTH))
        print(f"Processing {path} with shape {img.shape} to {preprocess_img.shape}")
        
        save_path = os.path.join(PREPROCESS_DATASET_PATH, path.split('/')[-1])
        cv2.imwrite(save_path, preprocess_img)
        shape_save_path = os.path.join(PREPROCESS_DATASET_PATH, path.split('/')[-1].split('.')[0]+"_shape.npy")
        np.save(shape_save_path, np.array(img.shape))
        
    store_path_on_board = '/userdata/val2017_preprocess/'
    with open('./coco_dataset_test_path.txt', 'w') as f:
        datalist = glob.glob(PREPROCESS_DATASET_PATH+"/*.jpg")
        for file in datalist:
            f.write(os.path.join(store_path_on_board, file.split('/')[-1])+'\n')
