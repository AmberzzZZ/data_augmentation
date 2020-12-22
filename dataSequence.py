import cv2
import numpy as np
import random
import math
import pandas as pd
from color import random_contrast, random_brightness, random_hsv, random_gamma
from coords import random_crop, random_expand, random_flip, random_rotate
from compose import mosaic
from keras.utils import Sequence
import json
import os


class dataSequence(Sequence):

    def __init__(self, img_dir, label_dir, input_shape, num_classes, anchors,
                 batch_size=1, shuffle=True, strides=[8,16,32],
                 negative_overlap=0.4, positive_overlap=0.5):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = anchors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.strides = strides
        self.negative_overlap = negative_overlap
        self.positive_overlap = positive_overlap

        self.train_lst = [i.replace('.png', '') for i in os.listdir(img_dir) if 'png' in i]
        self.full_lst = self.train_lst
        random.shuffle(self.full_lst)
        self.indices = np.arange(len(self.full_lst))

    def __len__(self):
        return math.ceil(len(self.full_lst) / float(self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size: (index+1)*self.batch_size]
        batch_lst = [self.full_lst[k] for k in batch_indices]
        x_batch, y_batch = self.data_generator(batch_lst)
        return x_batch, y_batch

    def on_epoch_end(self):
        random.shuffle(self.other_lst)
        self.full_lst = self.train_lst + self.other_lst[:20000]
        random.shuffle(self.full_lst)
        # if self.shuffle == True:
        #     np.random.shuffle(self.indices)

    def data_generator(self, batch_lst):
        input_shape = self.input_shape
        n_anchors = (self.anchors[0]).shape[0]
        n_classes = self.num_classes
        image_batch = np.zeros((self.batch_size, input_shape[0], input_shape[1], 3))
        y_true = [np.zeros((self.batch_size, input_shape[0]//s, input_shape[1]//s,
                            n_anchors, 4+n_classes+1)) for s in self.strides]

        target_shape = (input_shape[1], input_shape[0])
        for i, file_name in enumerate(batch_lst):
            try:
                img = cv2.imread(os.path.join(self.img_dir, file_name+'.png'), 1)
                if os.path.exists(os.path.join(self.label_dir, file_name+'.json')):
                    boxes = get_box(os.path.join(self.label_dir, file_name+'.json'))
                    boxes = boxes[:,:-1]
                    labels = boxes[:,-1:]
                else:
                    boxes = np.zeros((0))
                    labels = np.zeros((0))
                img, boxes, labels = aug_slice(img, boxes, labels, target_shape)
                if boxes.shape[0]:
                    boxes = np.concatenate([boxes, labels], axis=-1)
            except:
                img = np.zeros((target_shape)+(3,))
                boxes = np.zeros((0))

            if np.max(img)>1:
                img = img / 255.
            image_batch[i] = img
            if boxes.shape[0] <= 0:
                continue

            # arrange anchor
            input_shape = np.array(input_shape)
            boxes_xy = boxes[...,:2] * input_shape[::-1]
            boxes_wh = boxes[...,2:4] * input_shape[::-1]     # abs rela-origin-xcycwh
            boxes_abs = np.concatenate([boxes_xy, boxes_wh], axis=-1)

            for idx, s in enumerate(self.strides):
                anchors_wh = anchors[idx].reshape((1,-1,2))        # (1,N,2)
                grid_h, grid_w = input_shape[0]//s, input_shape[1]//s    # hw
                coords_x, coords_y = np.meshgrid(np.arange(grid_w),np.arange(grid_h))
                anchors_xy = (np.stack([coords_x, coords_y], axis=-1).reshape((-1,1,2))+0.5)*s   # (h*w,1,2)
                n_anchors_s = anchors_wh.shape[1]
                n_grids_s = anchors_xy.shape[0]
                anchors_wh = np.tile(anchors_wh, [n_grids_s, 1,1])
                anchors_xy = np.tile(anchors_xy, [1, n_anchors_s,1])
                anchors_abs = np.concatenate([anchors_xy,anchors_wh], axis=-1).reshape((-1,4))    # abs rela-origin-xcycwh, (h*w*n,4)
                iou = cal_iou(boxes_abs.copy(), anchors_abs.copy())
                # print("stride: ", s, "iou: ", np.max(iou, axis=-1))

                best_match_indices = np.argmax(iou, axis=-1)
                best_match_iou = np.max(iou, axis=-1)

                yt = np.zeros((anchors_abs.shape[0],4+num_classes+1))
                for b in range(boxes.shape[0]):
                    if best_match_iou[b] > self.positive_overlap:
                        yt[best_match_indices[b]][:4] = boxes[b,:4]
                        yt[best_match_indices[b]][4+int(boxes[b,-1])] = 1
                        yt[best_match_indices[b]][-1] = 1
                    elif best_match_iou[b] > self.negative_overlap:
                        yt[best_match_indices[b]][-1] = -1

                y_true[idx][i] = yt.reshape((grid_h, grid_w, n_anchors_s, -1))

        return [image_batch, *y_true], np.zeros(self.batch_size)


def aug_slice(img, boxes, labels, target_shape):   # target(w,h)
    # step1: crop/expand/resize
    if random.uniform(0, 1)>0.7:
        img, boxes, labels = random_expand(img, boxes, labels)
    elif random.uniform(0, 1)>0.7:
        img, boxes, labels = random_crop(img, boxes, labels)
    img = cv2.resize(img, target_shape)

    # step2: flip/rotate90
    if random.uniform(0, 1)>0.5:
        img, boxes, labels = random_flip(img, boxes, labels)
    else:
        img, boxes, labels = random_rotate(img, boxes, labels)

    # step3: hsv/brightness/contrast/gamma
    methods = [random_contrast, random_hsv, random_brightness, random_gamma, np.array]
    if random.uniform(0,1) > 0.5:
        transform = random.choice(methods)
        img = transform(img)

    return img, boxes, labels


def aug_mosaic(imgLst, boxLst, labelLst):
    return mosaic(imgLst, boxLst, labelLst, min_offset_x=0.4, min_offset_y=0.4)


def get_box(yolo_file):
    label_dict = {1:1, 0:0}
    f = open(yolo_file, 'r')
    boxes = json.loads(f.read())
    box_arr = []
    for b in boxes:
        if b['label'] not in label_dict.keys():
            continue
        box_arr.append([b['xc'],b['yc'],b['w'],b['h'], b['label']])
    return np.array(box_arr)


def get_anchors_yolo(anchors_path, n_anchors=3):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors, dtype=np.float32).reshape(-1,2)
    anchors_per_level = []
    for i in range(0, anchors.shape[0], n_anchors):
        anchors_per_level.append(anchors[i:i+n_anchors])
    return anchors_per_level


def cal_iou(boxes1, boxes2, epsilon=1e-5):

    # convert xcycwh to x1y1x2y2
    boxes1[:,0] -= 0.5*boxes1[:,2]
    boxes1[:,1] -= 0.5*boxes1[:,3]
    boxes1[:,2] += boxes1[:,0]
    boxes1[:,3] += boxes1[:,2]
    boxes2[:,0] -= 0.5*boxes2[:,2]
    boxes2[:,1] -= 0.5*boxes2[:,3]
    boxes2[:,2] += boxes2[:,0]
    boxes2[:,3] += boxes2[:,2]

    # boxes1: [N1,4], x1y1x2y2
    # boxes2: [N2,4], x1y1x2y2
    boxes1 = np.expand_dims(boxes1, axis=1)
    boxes2 = np.expand_dims(boxes2, axis=0)

    inter_mines = np.maximum(boxes1[...,:2], boxes2[...,:2])    # [N1,N2,2]
    inter_maxes = np.minimum(boxes1[...,2:], boxes2[...,2:])
    inter_wh = np.maximum(inter_maxes - inter_mines, 0.)
    inter_area = inter_wh[...,0] * inter_wh[...,1]

    box_area1 = (boxes1[...,2]-boxes1[...,0]) * (boxes1[...,3]-boxes1[...,1])
    box_area1 = np.tile(box_area1, [1,np.shape(boxes2)[0]])
    box_area2 = (boxes2[...,2]-boxes2[...,0]) * (boxes2[...,3]-boxes2[...,1])
    box_area2 = np.tile(box_area2, [np.shape(boxes1)[0],1])

    iou = inter_area / (box_area1 + box_area2 - inter_area + epsilon)

    return iou


if __name__ == '__main__':

    data_dir = "data"
    label_dir = "data"
    batch_size = 1
    input_shape = (512,512)   # hw
    num_classes = 15
    strides = [8, 16, 32]
    n_anchors = 3         # for each level
    anchors = get_anchors_yolo("data/anchors.txt", n_anchors)

    generator = dataSequence(data_dir, label_dir, input_shape, num_classes, anchors,
                             batch_size, shuffle=True, strides=strides,
                             negative_overlap=0.4, positive_overlap=0.5)

    stride_cnt = [0 for i in strides]
    wrong_cnt = 0
    for idx, [x_batch, y_batch] in enumerate(generator):
        print(idx)
        image_batch = x_batch[0]
        print("img input: ", image_batch.shape, np.max(image_batch))
        if np.max(image_batch)<0.001:
            wrong_cnt += 1
            continue
        y_true = x_batch[1:]
        print("gt input: ", [np.unique(i[...,-1]) for i in y_true])

        # vis
        for i, s in enumerate(strides):
            gt = y_true[i][0]
            cls_prob = np.max(gt[:,:,:,4:-1], axis=-1)
            coords = np.where(cls_prob>0.5)
            print("stride: ", s, "n_obj: ", len(coords[0]))
            stride_cnt[i] += len(coords[0])

        if idx>100:
            break
    print(stride_cnt)









