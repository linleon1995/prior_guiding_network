import tensorflow as tf

def spatial_transfom_exp(image, angle=None, translations=None, interpolation="BILINEAR"):
    """translations: [dx, dy]"""
    h, w = image.get_shape().as_list()[1:3]
    # Rotate imagewith given angle
    if angle is not None:
        image = tf.contrib.image.rotate(image,
                                        angle,
                                        interpolation=interpolation
                                        )
        image.set_shape([None,h,w,1])
    # Translate imagewith given shifting pixels
    if translations is not None:
        image = tf.contrib.image.translate(image,
                                        translations,
                                        interpolation=interpolation)
        image.set_shape([None,h,w,1])
    return image