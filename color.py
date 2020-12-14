import cv2
import numpy as np
import random


# [0,255] img

def random_brightness(img, bright_vari=32):
    bright_add = np.random.uniform(-bright_vari, bright_vari, img.shape)
    img = img + bright_add
    img[img>255] = 255
    img[img<0] = 0
    return np.uint8(img)


# hue, saturation, value
def random_hsv(img, hue_vari=10, sat_vari=0.5, val_vari=0.5):
    hue_add = random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + random.uniform(-val_vari, val_vari)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    if random.uniform(0, 1)>0.5:
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + hue_add
    if random.uniform(0, 1)>0.5:
        img_hsv[:, :, 1] *= sat_mult
    if random.uniform(0, 1)>0.5:
        img_hsv[:, :, 2] *= val_mult
    img_hsv = (img_hsv - np.min(img_hsv)) / (np.max(img_hsv) - np.min(img_hsv))
    img = cv2.cvtColor(np.float32(img_hsv), cv2.COLOR_HSV2BGR)
    return np.uint8(img*255)


def random_lightingnoise(img):
    perms = ((0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0))
    swap = random.choice(perms)
    img = img[:,:,swap]
    return img


def random_gamma(img, gamma_vari=1.5):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


if __name__ == '__main__':

    img = cv2.imread("data/tux_hacking.png", 1)

    for i in range(10):
        # img2 = random_brightness(img)
        img2 = random_hsv(img)
        # img2 = random_gamma(img)
        # img2 = random_lightingnoise(img)
        cv2.imshow("tmp2", img2)
        cv2.waitKey(0)





