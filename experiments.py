import tensorflow as tf

def spatial_transfom_exp(image, angle=None, translations=None, interpolation="BILINEAR"):
    """translations: [dx, dy]"""
    n, h, w, c = image.get_shape().as_list()
    # Rotate imagewith given angle
    if angle is not None:
        image = tf.contrib.image.rotate(image,
                                        angle,
                                        interpolation=interpolation
                                        )
        image.set_shape([n,h,w,c])
    # Translate imagewith given shifting pixels
    if translations is not None:
        image = tf.contrib.image.translate(image,
                                        translations,
                                        interpolation=interpolation)
        image.set_shape([n,h,w,c])
    return image