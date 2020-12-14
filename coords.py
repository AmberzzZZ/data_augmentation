import cv2
import numpy as np
import random
import json
import math


# [0,255] img, [h,w,c]
# [0,1] normed xcycwh, [N,4]


def random_expand(img, boxes, labels):
    img = img.copy()
    boxes = boxes.copy()
    labels = labels.copy()
    h, w, c = img.shape
    gaps = []
    for i in range(4):
        gaps.append(random.randint(int(min(w,h)*0.5), int(min(w,h)*1.5)))
    img = cv2.copyMakeBorder(img, gaps[0], gaps[1], gaps[2], gaps[3], cv2.BORDER_CONSTANT,
                             value=(np.mean(img), np.mean(img), np.mean(img)))
    img = cv2.resize(img, dsize=(w,h))
    new_xc = (boxes[...,0]*w + gaps[2]) / (w + gaps[2] + gaps[3])
    new_yc = (boxes[...,1]*h + gaps[0]) / (h + gaps[0] + gaps[1])
    new_w = boxes[...,2]*w / (w + gaps[2] + gaps[3])
    new_h = boxes[...,3]*h / (h + gaps[0] + gaps[1])
    return img, np.stack([new_xc, new_yc, new_w, new_h], axis=-1), labels


def random_crop(img, boxes, labels, iou_thresh=0.5):
    img = img.copy()
    boxes = boxes.copy()
    labels = labels.copy()
    h, w, c = img.shape
    boxes_x1y1x2y2 = np.zeros_like(boxes)
    boxes_x1y1x2y2[...,0] = boxes[...,0]-boxes[...,2]/2
    boxes_x1y1x2y2[...,1] = boxes[...,1]-boxes[...,3]/2
    boxes_x1y1x2y2[...,2] = boxes[...,0]+boxes[...,2]/2
    boxes_x1y1x2y2[...,3] = boxes[...,1]+boxes[...,3]/2
    for _ in range(50):
        new_w = random.randint(int(w*0.3), w)
        new_h = random.randint(int(h*0.3), h)
        if new_h/new_w>2 or new_h/new_w<0.5:
            continue
        new_x1 = random.randint(0, w-new_w)
        new_y1 = random.randint(0, h-new_h)
        boxes_xcyc = boxes[...,:2] * [[w, h]]
        center_in = (boxes_xcyc>[[new_x1,new_y1]]) & (boxes_xcyc<[[new_x1+new_w,new_y1+new_h]])
        center_in = center_in[:,0] & center_in[:,1]
        if not center_in.any():
            continue
        new_box = np.array([[new_x1/w, new_y1/h, (new_x1+new_w)/w, (new_y1+new_h)/h]])
        iou = cal_iou(boxes_x1y1x2y2, new_box)
        if np.max(iou) < iou_thresh:
            continue
        img = img[new_y1:new_y1+new_h, new_x1:new_x1+new_w]
        boxes_x1y1x2y2 = boxes_x1y1x2y2[center_in]
        labels = labels[center_in]
        boxes_x1 = np.maximum(boxes_x1y1x2y2[:,0]*w, new_x1) - new_x1
        boxes_y1 = np.maximum(boxes_x1y1x2y2[:,1]*h, new_y1) - new_y1
        boxes_x2 = np.minimum(boxes_x1y1x2y2[:,2]*w, new_x1+new_w) - new_x1
        boxes_y2 = np.minimum(boxes_x1y1x2y2[:,3]*h, new_y1+new_h) - new_y1
        boxes_xc = (boxes_x1+boxes_x2) / 2 / new_w
        boxes_yc = (boxes_y1+boxes_y2) / 2 / new_h
        boxes_w = (boxes_x2 - boxes_x1) / new_w
        boxes_h = (boxes_y2 - boxes_y1) / new_h
        boxes = np.stack([boxes_xc, boxes_yc, boxes_w, boxes_h], axis=-1)

        return img, boxes, labels

    return img, boxes, labels


def random_flip(img, boxes, labels):
    img = img.copy()
    boxes = boxes.copy()
    labels = labels.copy()
    if random.uniform(0, 1)>0.5:
        img = img[:,::-1,:].copy()
        boxes[:,0] = 1 - boxes[:,0]
    else:
        img = img[::-1,:,:].copy()
        boxes[:,1] = 1 - boxes[:,1]
    return img, boxes, labels


def random_rotate(img, boxes, labels):
    img = img.copy()
    boxes = boxes.copy()
    labels = labels.copy()
    h, w, c = img.shape
    # 90, 180, 270
    rotate_angle = random.choice([-math.pi/2, math.pi/2, math.pi])
    tl = boxes[:,:2]*[w,h] + np.stack([-boxes[:,2]*w/2, -boxes[:,3]*h/2], axis=-1)
    tr = boxes[:,:2]*[w,h] + np.stack([boxes[:,2]*w/2, -boxes[:,3]*h/2], axis=-1)
    bl = boxes[:,:2]*[w,h] + np.stack([-boxes[:,2]*w/2, boxes[:,3]*h/2], axis=-1)
    br = boxes[:,:2]*[w,h] + np.stack([boxes[:,2]*w/2, boxes[:,3]*h/2], axis=-1)
    n_points = tl.shape[0]
    points = list(tl) + list(tr) + list(bl) + list(br)
    img, points = rotate_img(rotate_angle, img, points)
    new_tl = np.array(points[:n_points])
    new_tr = np.array(points[n_points:n_points*2])
    new_bl = np.array(points[n_points*2:n_points*3])
    new_br = np.array(points[n_points*3:])
    left = np.minimum(new_tl[:,0], new_bl[:,0])
    right = np.maximum(new_tr[:,0], new_br[:,0])
    top = np.minimum(new_tl[:,1], new_tr[:,1])
    bottom = np.maximum(new_bl[:,1], new_br[:,1])
    new_h = int(w*math.fabs(math.sin(rotate_angle)) + h*math.fabs(math.cos(rotate_angle)))
    new_w = int(h*math.fabs(math.sin(rotate_angle)) + w*math.fabs(math.cos(rotate_angle)))
    xc = (left + right) / 2 / new_w
    yc = (top + bottom) / 2 / new_h
    w = np.abs(right - left) / new_w
    h = np.abs(bottom - top) / new_h
    boxes = np.stack([xc, yc, w, h], axis=-1)
    return img, boxes, labels


def rotate_img(angle, img, points=[], interpolation=cv2.INTER_LINEAR):
    h, w, c = img.shape
    rotataMat = cv2.getRotationMatrix2D((w/2-0.5, h/2-0.5), math.degrees(angle), 1)
    new_h = int(w*math.fabs(math.sin(angle)) + h*math.fabs(math.cos(angle)))
    new_w = int(h*math.fabs(math.sin(angle)) + w*math.fabs(math.cos(angle)))
    rotataMat[0, 2] += (new_w - w) / 2
    rotataMat[1, 2] += (new_h - h) / 2
    # img
    rotate_img = cv2.warpAffine(img, rotataMat, (new_w, new_h), flags=interpolation, borderValue=(0,0,0))
    # points
    rotated_points = []
    for point in points:
        point = rotataMat.dot([[point[0]], [point[1]], [1]])
        rotated_points.append((int(point[0]), int(point[1])))
    return rotate_img, rotated_points


def cal_iou(boxes1, boxes2, epsilon=1e-5):
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

    img = cv2.imread("data/tux_hacking.png", 1)
    with open('data/tux_hacking.json', 'r') as f:
        label = json.loads(f.read())

    boxes = []
    labels = []
    for bbox in label:
        boxes.append([bbox['xc'], bbox['yc'], bbox['w'], bbox['h']])
        labels.append(bbox['label'])
    boxes = np.array(boxes)
    labels = np.array(labels)

    for i in range(10):
        img2, boxes2, _ = random_expand(img, boxes, labels)
        # img2, boxes2, _ = random_crop(img, boxes, labels)
        # img2, boxes2, _ = random_flip(img, boxes, labels)
        # img2, boxes2, _ = random_rotate(img, boxes, labels)
        h, w, c = img2.shape
        for i in range(boxes2.shape[0]):
            bbox = boxes2[i]
            cv2.rectangle(img2, (int((bbox[0]-bbox[2]/2)*w), int((bbox[1]-bbox[3]/2)*h)),
                          (int((bbox[0]+bbox[2]/2)*w), int((bbox[1]+bbox[3]/2)*h)), (0,0,255), 2)

        cv2.imshow("tmp2", img2)
        cv2.waitKey(0)
        # cv2.imwrite("random_rotate.png", img2)
