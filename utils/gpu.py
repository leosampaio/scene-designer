import tensorflow as tf
import logging


def setup_gpu(gpu_ids):

    logger = tf.get_logger()
    logger.setLevel(logging.INFO)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            sel_gpus = [gpus[g] for g in gpu_ids]
            tf.config.set_visible_devices(sel_gpus, 'GPU')
            for g in sel_gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            # visible devices must be set before GPUs have been initialized
            print(e)
