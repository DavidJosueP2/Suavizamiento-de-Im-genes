import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def normalizar_visual(imagen):
    """Normaliza una imagen de punto flotante a rango 0-255 uint8."""
    minimo = imagen.min()
    maximo = imagen.max()
    if maximo == minimo:
        return np.zeros_like(imagen, dtype=np.uint8)
    normalizada = (imagen - minimo) / (maximo - minimo) * 255.0
    return np.clip(normalizada, 0, 255).astype(np.uint8)


def aplicar_por_canal(imagen, funcion):
    """Aplica una función a cada canal de una imagen RGB o a la imagen en grises."""
    if imagen.ndim == 2:
        return funcion(imagen)
    canales = [funcion(imagen[:, :, i]) for i in range(imagen.shape[2])]
    return np.stack(canales, axis=2).astype(np.uint8)


def convertir_grises(imagen):
    """Convierte una imagen RGB a escala de grises usando luminancia."""
    if imagen.ndim == 2:
        return imagen.astype(np.uint8)
    gris = (
        0.299 * imagen[:, :, 0].astype(np.float32)
        + 0.587 * imagen[:, :, 1].astype(np.float32)
        + 0.114 * imagen[:, :, 2].astype(np.float32)
    )
    return np.clip(np.rint(gris), 0, 255).astype(np.uint8)


def aplicar_padding_replicado(imagen, tamano_mascara):
    """Aplica padding replicando bordes para convolución."""
    radio = tamano_mascara // 2
    alto, ancho = imagen.shape
    padded = np.zeros((alto + 2 * radio, ancho + 2 * radio), dtype=imagen.dtype)
    padded[radio : radio + alto, radio : radio + ancho] = imagen
    # Replicar bordes
    padded[:radio, radio : radio + ancho] = imagen[0:1, :]
    padded[radio + alto :, radio : radio + ancho] = imagen[alto - 1 : alto, :]
    padded[:, :radio] = padded[:, radio : radio + 1]
    padded[:, radio + ancho :] = padded[:, radio + ancho - 1 : radio + ancho]
    return padded


def ventanas_aplanadas(imagen, tamano_mascara):
    """Genera ventanas deslizantes aplanadas para filtros estadísticos."""
    padded = aplicar_padding_replicado(imagen, tamano_mascara)
    ventanas = sliding_window_view(padded, (tamano_mascara, tamano_mascara))
    return ventanas.reshape(
        imagen.shape[0], imagen.shape[1], tamano_mascara * tamano_mascara
    )


def convolucion_cruda(imagen, kernel):
    """Convolución 2D sin normalización, retorna float."""
    kernel = np.array(kernel, dtype=np.float32)
    k = kernel.shape[0]
    padded = aplicar_padding_replicado(imagen, k)
    ventanas = sliding_window_view(padded, (k, k))
    return np.sum(ventanas * kernel, axis=(2, 3))


def convolucion_normalizada(imagen, kernel):
    """Convolución 2D con normalización visual a uint8."""
    resultado = convolucion_cruda(imagen, kernel)
    return normalizar_visual(resultado)
