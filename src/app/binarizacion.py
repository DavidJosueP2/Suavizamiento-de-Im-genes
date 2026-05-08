import numpy as np


def binarizar_imagen(imagen, umbral):
    return np.where(imagen >= int(umbral), 255, 0).astype(np.uint8)
