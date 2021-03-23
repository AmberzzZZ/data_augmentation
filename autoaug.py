from PIL import Image, ImageEnhance, ImageOps, ImageTransform
import numpy as np
import random
import inspect


# [0,255] img

_MAX_LEVEL = 10.


def cutout(img, pad_size, replace=0.):
    # applies a (2*pad_size x 2*pad_size) mask
    img = img.copy()
    h, w = img.shape[:2]
    center_h, center_w = random.randint(0, h), random.randint(0, w)
    lower_pad = max(0, center_h - pad_size)
    upper_pad = max(0, h - center_h - pad_size)
    left_pad = max(0, center_w - pad_size)
    right_pad = max(0, w - center_w - pad_size)
    pad_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    cutout_shape = [h-lower_pad-upper_pad, w-left_pad-right_pad]
    mask = np.pad(np.zeros((cutout_shape)), pad_dims, mode='constant', constant_values=1)
    mask = np.expand_dims(mask, -1)
    mask = np.tile(mask, [1, 1, 3])
    img[mask==0] = replace
    return img


def solarize(img, threshold=128):
    img = ImageOps.solarize(Image.fromarray(img), threshold=threshold)
    return np.array(img)


def solarize_add(img, addition=0, threshold=128):
    img = img.copy().astype(np.int64)
    added_img = img + addition
    added_img[added_img>255] = 255
    added_img[added_img<0] = 0
    img[img<threshold] = added_img[img<threshold]
    return np.uint8(img)


def color(img, factor):
    img = ImageEnhance.Color(Image.fromarray(img)).enhance(factor)
    return np.array(img)


def contrast(img, factor):
    img = ImageEnhance.Contrast(Image.fromarray(img)).enhance(factor)
    return np.array(img)


def brightness(img, factor):
    img = ImageEnhance.Brightness(Image.fromarray(img)).enhance(factor)
    return np.array(img)


def sharpness(img, factor):
    img = ImageEnhance.Sharpness(Image.fromarray(img)).enhance(factor)
    return np.array(img)


def posterize(img, bits):
    img = ImageOps.posterize(Image.fromarray(img), bits=bits)
    return np.array(img)


def rotate(img, degrees, replace):
    img = Image.fromarray(img).rotate(degrees, fillcolor=replace)
    return np.array(img)


def translate_x(img, pixels, replace):
    img = Image.fromarray(img).rotate(0, translate=(pixels,0), fillcolor=replace)
    return np.array(img)


def translate_y(img, pixels, replace):
    img = Image.fromarray(img).rotate(0, translate=(0,pixels), fillcolor=replace)
    return np.array(img)


def shear_x(img, level, replace):
    pil_img = Image.fromarray(img)
    img = pil_img.transform(pil_img.size, Image.AFFINE, (1., level, 0., 0., 1., 0., 0., 0.))
    return np.array(img)


def shear_y(img, level, replace):
    pil_img = Image.fromarray(img)
    img = pil_img.transform(pil_img.size, Image.AFFINE, (1., 0, 0., level, 1., 0., 0., 0.))
    return np.array(img)


def autocontrast(img):
    img = ImageOps.autocontrast(Image.fromarray(img))
    return np.array(img)


def equalize(img):
    img = ImageOps.equalize(Image.fromarray(img))
    return np.array(img)


def invert(img):
    img = img.copy()
    return 255-img


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
}


def _parse_policy_info(name, prob, level, replace_value, additional_hyparams):
    func = NAME_TO_FUNC[name]
    args = level_to_arg(additional_hyparams)[name](level)

    if 'prob' in inspect.getfullargspec(func)[0]:
        args = tuple([prob] + list(args))

    if 'replace' in inspect.getfullargspec(func)[0]:
        assert 'replace' == inspect.getfullargspec(func)[0][-1]
        args = tuple(list(args) + [replace_value])

    return (func, prob, args)


def _randomly_negate_tensor(value):
    if random.uniform(0, 1)>0.5:
        return -value


def _rotate_level_to_arg(level):
    level = (level/_MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level)
    return (level,)


def _shrink_level_to_arg(level):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return (1.0,)  # if level is zero, do not shrink the image
    # Maximum shrinking ratio is 2.9.
    level = 2. / (_MAX_LEVEL / level) + 0.9
    return (level,)


def _enhance_level_to_arg(level):
    return ((level/_MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level):
    level = (level/_MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level,)


def _translate_level_to_arg(level, translate_const):
    level = (level/_MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level,)


def level_to_arg(hparams):
    return {
        'AutoContrast': lambda level: (),
        'Equalize': lambda level: (),
        'Invert': lambda level: (),
        'Rotate': _rotate_level_to_arg,
        'Posterize': lambda level: (int((level/_MAX_LEVEL) * 4),),
        'Solarize': lambda level: (int((level/_MAX_LEVEL) * 256),),
        'SolarizeAdd': lambda level: (int((level/_MAX_LEVEL) * 110),),
        'Color': _enhance_level_to_arg,
        'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg,
        'Sharpness': _enhance_level_to_arg,
        'ShearX': _shear_level_to_arg,
        'ShearY': _shear_level_to_arg,
        'Cutout': lambda level: (int((level/_MAX_LEVEL) * hparams.cutout_const),),
        # pylint:disable=g-long-lambda
        'TranslateX': lambda level: _translate_level_to_arg(
            level, hparams.translate_const),
        'TranslateY': lambda level: _translate_level_to_arg(
            level, hparams.translate_const),
        # pylint:enable=g-long-lambda
    }


def distort_image_with_randaugment(image, num_layers, magnitude):
    replace_value = [128] * 3

    class hyparams:
        cutout_const = 40
        translate_const = 100
    additional_hyparams = hyparams()

    available_ops = [
        'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
        'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd']

    for layer_num in range(num_layers):
        op_to_select = random.randint(0, len(available_ops))
        random_magnitude = float(magnitude)
        # do aug
        prob = random.uniform(0.2, 0.8)
        func, _, args = _parse_policy_info(available_ops[op_to_select], prob, random_magnitude,
                                           replace_value, additional_hyparams)
        image = func(image, *args)
    return image


if __name__ == '__main__':

    img = Image.open("/Users/amber/Downloads/tux_hacking.jpg")
    img = np.array(img)

    # img = cutout(img, pad_size=40)
    # img = solarize_add(img, addition=33, threshold=128)
    # img = color(img, factor=0.2)
    # img = contrast(img, factor=2.2)
    # img = rotate(img, 60, replace=(128,128,128))
    # img = translate_x(img, 60, replace=(128,128,128))
    # img = translate_y(img, 60, replace=(128,128,128))
    # img = shear_x(img, 0.1, replace=(128,128,128))
    # img = shear_y(img, 0.1, replace=(128,128,128))
    # img = autocontrast(img)
    # img = brightness(img, factor=0.2)
    # img = sharpness(img, factor=10.2)
    # img = equalize(img)
    # img = invert(img)

    img = distort_image_with_randaugment(img, 3, 10)
    Image.fromarray(img).show()





