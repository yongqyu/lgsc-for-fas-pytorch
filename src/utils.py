import tensorflow as tf

def make_grid(x):
    t = tf.unstack(x, axis=0)
    image = tf.concat(t, axis=0)
    return image

class GridMaker:
    def __init__(self):
        pass

    def __call__(self, images, cues):
        b, c, h, w = images.shape
        images_min = tf.reduce_min(tf.reshape(images, (b, -1)), axis=1)[:, None]
        images_max = tf.reduce_max(tf.reshape(images, (b, -1)), axis=1)[:, None]
        images = (tf.reshape(images, (b, -1)) - images_min) / (images_max - images_min)
        images = tf.reshape(images, (b, c, h, w))

        b, c, h, w = cues.shape
        cues_min = tf.reduce_min(tf.reshape(cues, (b, -1)), axis=1)[:, None]
        cues_max = tf.reduce_max(tf.reshape(cues, (b, -1)), axis=1)[:, None]
        cues = (tf.reshape(cues, (b, -1)) - cues_min) / (cues_max - cues_min)
        cues = tf.reshape(cues, (b, c, h, w))

        return make_grid(images), make_grid(cues)
