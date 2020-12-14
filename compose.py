import cv2
import numpy as np
import random
import math
import json


# [0,255] img, [h,w,c]
# [0,1] normed xcycwh, [N,4]

def cutMix(img1, labels1, img2, labels2):
    img1 = img1.copy()
    img2 = img2.copy()
    # cut area
    alpha = random.uniform(0, 1)
    lam = np.random.beta(alpha, alpha)
    h, w, c = img1.shape
    x1, x2, y1, y2 = rand_box(h, w, lam)
    img1[y1:y2, x1:x2, :] = img2[y1:y2, x1:x2, :]
    # reweight
    lam = (x2 - x1) * (y2 - y1) / (h * w)
    label = np.float32(labels1) * (1-lam) + np.float32(labels2) * lam
    return img1, label


def mosaic(imgs_lst, boxes_lst, labels_lst, min_offset_x=0.4, min_offset_y=0.4):
    h, w, c = imgs_lst[0].shape
    canvas = np.zeros((int(h/(1-min_offset_y)), int(w/(1-min_offset_x)), 3), dtype=np.uint8)
    canvas_h, canvas_w, c = canvas.shape
    place_x = [0,0,int(canvas_w*min_offset_x),int(canvas_w*min_offset_x)]
    place_y = [0,int(canvas_h*min_offset_y),int(canvas_h*min_offset_y),0]
    canvas_lst = []
    canvas_boxes_lst = []
    canvas_labels_lst = []
    for i in range(4):
        tmp = canvas.copy()
        x1 = place_x[i]
        y1 = place_y[i]
        tmp[y1:y1+h, x1:x1+w, :] = imgs_lst[i].copy()
        canvas_lst.append(tmp)
        if boxes_lst[i].shape[0]>0:
            # abs
            boxes_xcyc = boxes_lst[i][:,:2] * [w,h] + [x1,y1]
            boxes_wh = boxes_lst[i][:,2:] * [w,h]
            canvas_boxes_lst.append(np.concatenate([boxes_xcyc, boxes_wh], axis=-1))
        else:
            canvas_boxes_lst.append(None)
        canvas_labels_lst.append(labels_lst[i])
    cutx = random.randint(int(canvas_w*min_offset_x), int(canvas_w*(1-min_offset_x)))
    cuty = random.randint(int(canvas_h*min_offset_y), int(canvas_h*(1-min_offset_y)))
    canvas[:cuty, :cutx, :] = canvas_lst[0][:cuty, :cutx, :]   # left-top
    canvas[cuty:, :cutx, :] = canvas_lst[1][cuty:, :cutx, :]   # left-bottom
    canvas[cuty:, cutx:, :] = canvas_lst[2][cuty:, cutx:, :]   # right-bottom
    canvas[:cuty, cutx:, :] = canvas_lst[3][:cuty, cutx:, :]   # right-top
    canvas_boxes, canvas_labels = merge_boxes(canvas_boxes_lst, canvas_labels_lst, cutx, cuty, canvas_h, canvas_w)
    return canvas, canvas_boxes, canvas_labels


def merge_boxes(boxes_lst, labels_lst, cutx, cuty, canvas_h, canvas_w):
    canvas_boxes = []
    canvas_labels = []
    box_limit = [[0,0,cutx,cuty],
                 [0,cuty,cutx,canvas_h],
                 [cutx,cuty,canvas_w,canvas_h],
                 [cutx,0,canvas_w,cuty]]

    for i in range(4):
        if boxes_lst[i] is None:
            continue
        boxes = boxes_lst[i]
        boxes_x1y1 = boxes[:,:2]-boxes[:,2:]/2
        boxes_x2y2 = boxes[:,:2]+boxes[:,2:]/2
        x1,y1,x2,y2 = box_limit[i]
        boxes_x1 = np.maximum(boxes_x1y1[:,0], x1)
        boxes_y1 = np.maximum(boxes_x1y1[:,1], y1)
        boxes_x2 = np.minimum(boxes_x2y2[:,0], x2)
        boxes_y2 = np.minimum(boxes_x2y2[:,1], y2)
        # to norm
        boxes_xcyc = np.stack([boxes_x1+boxes_x2, boxes_y1+boxes_y2], axis=-1) / 2 / [canvas_w, canvas_h]
        boxes_wh = np.stack([boxes_x2-boxes_x1, boxes_y2-boxes_y1], axis=-1) / [canvas_w, canvas_h]
        boxes = np.concatenate([boxes_xcyc,boxes_wh], axis=-1)
        canvas_boxes.append(boxes)
        canvas_labels.append(labels_lst[i])
    boxes_lst = np.concatenate(canvas_boxes, axis=0)
    return boxes_lst, labels_lst


def rand_box(h, w, lam):
    xc = random.randint(0, w)
    yc = random.randint(0, h)
    box_w = int(w*math.sqrt(1-lam))
    box_h = int(h*math.sqrt(1-lam))

    box_x1 = np.clip(xc-box_w//2, 0, box_w)
    box_x2 = np.clip(xc+box_w//2, 0, box_w)
    box_y1 = np.clip(yc-box_h//2, 0, box_h)
    box_y2 = np.clip(yc+box_h//2, 0, box_h)

    return box_x1, box_x2, box_y1, box_y2


if __name__ == '__main__':

    img1 = cv2.imread("data/tux_hacking.png", 1)
    img2 = cv2.imread("data/bialetti.png", 1)

    # test cutmix
    for i in range(10):
        img, label = cutMix(img1, [0,1], img2, [1,0])
        print(label)
        cv2.imshow("tmp", img)
        cv2.waitKey(0)
        cv2.imwrite('cutmix.png', img)


    with open('data/tux_hacking.json', 'r') as f:
        label = json.loads(f.read())
    boxes = []
    labels = []
    for bbox in label:
        boxes.append([bbox['xc'], bbox['yc'], bbox['w'], bbox['h']])
        labels.append(bbox['label'])
    boxes = np.array(boxes)
    labels = np.array(labels)

    # # test mosaic
    # for i in range(10):
    #     empty_boxes = np.zeros((0))
    #     img, boxes_, _ = mosaic([img2,img1,img2,img1],
    #                             [empty_boxes, boxes, empty_boxes, boxes],
    #                             [[], [], [], labels])

    #     h, w, c = img.shape
    #     for i in range(boxes_.shape[0]):
    #         bbox = boxes_[i]
    #         cv2.rectangle(img, (int((bbox[0]-bbox[2]/2)*w), int((bbox[1]-bbox[3]/2)*h)),
    #                       (int((bbox[0]+bbox[2]/2)*w), int((bbox[1]+bbox[3]/2)*h)), (0,0,255), 2)

    #     cv2.imshow("tmp2", img)
    #     cv2.waitKey(0)
    #     cv2.imwrite('mosaic.png', img)







