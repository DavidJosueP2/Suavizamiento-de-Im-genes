import numpy as np
from src.app.utils import normalizar_visual, convertir_grises, convolucion_cruda


def obtener_kernels_sobel():
    return {
        "0": [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        "45": [
            [0, 1, 2],
            [-1, 0, 1],
            [-2, -1, 0],
        ],
        "90": [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        "135": [
            [2, 1, 0],
            [1, 0, -1],
            [0, -1, -2],
        ],
    }


def obtener_kernels_prewitt():
    return {
        "0": [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ],
        "45": [
            [0, 1, 1],
            [-1, 0, 1],
            [-1, -1, 0],
        ],
        "90": [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1],
        ],
        "135": [
            [1, 1, 0],
            [1, 0, -1],
            [0, -1, -1],
        ],
    }


def reconocer_bordes_gradiente_direccional(imagen, metodo="Prewitt"):
    gris = convertir_grises(imagen)

    if metodo == "Sobel":
        kernels = obtener_kernels_sobel()
    else:
        kernels = obtener_kernels_prewitt()

    respuestas = {}

    for direccion, kernel in kernels.items():
        respuestas[direccion] = convolucion_cruda(gris, kernel)

    magnitud = np.zeros_like(gris, dtype=np.float32)

    for respuesta in respuestas.values():
        magnitud += respuesta.astype(np.float32) ** 2

    magnitud = np.sqrt(magnitud)

    resultados = {
        direccion: normalizar_visual(resp) for direccion, resp in respuestas.items()
    }
    resultados["magnitud"] = normalizar_visual(magnitud)
    return resultados


def _reconocer_bordes_gradiente(imagen, metodo="Prewitt"):
    gris = convertir_grises(imagen)
    if metodo == "Sobel":
        gx = convolucion_cruda(gris, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        gy = convolucion_cruda(gris, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    else:  # Prewitt
        gx = convolucion_cruda(gris, [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        gy = convolucion_cruda(gris, [[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    magnitud = np.sqrt(gx * gx + gy * gy)
    return (normalizar_visual(gx), normalizar_visual(gy), normalizar_visual(magnitud))
