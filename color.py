import cv2
import numpy as np
import random


# [0,255] img

def random_noise(img):
    img = img.copy()
    bright_vari=32
    bright_add = np.random.uniform(-bright_vari, bright_vari, img.shape)
    img = img + bright_add
    img[img>255] = 255
    img[img<0] = 0
    return np.uint8(img)


def random_brightness(img):
    img = img.copy()
    bright_vari=50
    bright_vari = random.uniform(-bright_vari, bright_vari)
    img = img + bright_vari
    img[img>255] = 255
    img[img<0] = 0
    return np.uint8(img)


def random_contrast(img):
    img = img.copy()
    contrast_vari = random.uniform(0.5, 1.5)
    img = img * contrast_vari
    img[img>255] = 255
    img[img<0] = 0
    return np.uint8(img)


# hue, saturation, value
def random_hsv(img):
    img = img.copy().astype(np.float32) / 255.
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_add = random.uniform(-360, 360)
    sat_mult = random.uniform(0.5, 1.5)
    if random.uniform(0, 1)>0.5:
        img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0] + hue_add, 0, 360)
    if random.uniform(0, 1)>0.5:
        img_hsv[:, :, 1] *= sat_mult
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return np.uint8(img*255)


def random_lightingnoise(img):
    img = img.copy()
    perms = ((0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0))
    swap = random.choice(perms)
    img = img[:,:,swap]
    return img


def random_gamma(img):
    img = img.copy()
    gamma = random.choice([0.4, 0.5, 0.8, 1.0, 1.2, 1.5])
    inv_gamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** inv_gamma) * 255)
    table = np.uint8(table)
    return cv2.LUT(img, table)


if __name__ == '__main__':

    img = cv2.imread("data/tux_hacking.png", 1)

    for i in range(10):
        # img2 = random_noise(img)
        # img2 = random_brightness(img)
        # img2 = random_contrast(img)
        # img2 = random_hsv(img)
        img2 = random_gamma(img)
        # img2 = random_lightingnoise(img)
        cv2.imshow("tmp2", img2)
        cv2.waitKey(0)
        cv2.imwrite("random_gamma.png", img2)





