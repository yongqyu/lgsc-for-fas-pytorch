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

def set_gmem_growth(gpus=None, yn=True):
    if gpus is None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, yn)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def read_label(path, anti_word=None, real_word=None):
    if anti_word != None:
        target = 1 if anti_word in path else 0
    elif real_word != None:
        target = 0 if real_word in path else 1
    else:
        target = 0 if int(path.split('.')[0].split('_')[-1]) == 0 else 1
    return target
