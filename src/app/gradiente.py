import numpy as np
from src.app.utils import normalizar_visual, convertir_grises, convolucion_cruda


def reconocer_bordes_gradiente(imagen, metodo="Prewitt"):
    gris = convertir_grises(imagen)
    if metodo == "Sobel":
        gx = convolucion_cruda(gris, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        gy = convolucion_cruda(gris, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    else:  # Prewitt
        gx = convolucion_cruda(gris, [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        gy = convolucion_cruda(gris, [[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    magnitud = np.sqrt(gx * gx + gy * gy)
    return (normalizar_visual(gx), normalizar_visual(gy), normalizar_visual(magnitud))
