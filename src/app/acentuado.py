import numpy as np
from src.app.utils import (
    normalizar_visual,
    aplicar_por_canal,
    convertir_grises,
    convolucion_cruda,
)


def laplaciano_espacial(imagen):
    kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    respuesta = convolucion_cruda(imagen, kernel)
    realzada = imagen.astype(np.float32) + respuesta
    return np.clip(realzada, 0, 255).astype(np.uint8)


def laplaciano_8_vecinos(imagen):
    kernel = [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ]
    respuesta = convolucion_cruda(imagen, kernel)
    realzada = imagen.astype(np.float32) + respuesta
    return np.clip(realzada, 0, 255).astype(np.uint8)


def crear_mascara_pasa_alto(forma, radio):
    from src.app.suavizado_frecuencia import crear_mascara_pasa_bajo

    return 1.0 - crear_mascara_pasa_bajo(forma, radio)


def crear_mascara_pasa_alto_gaussiana(forma, sigma):
    alto, ancho = forma[:2]
    cy, cx = alto // 2, ancho // 2
    y, x = np.ogrid[:alto, :ancho]
    distancia2 = (y - cy) ** 2 + (x - cx) ** 2
    mascara = 1.0 - np.exp(-distancia2 / (2 * (sigma**2)))
    return mascara.astype(np.float64)


def reconstruir_respuesta_ifft(transformada_filtrada):
    transformada_descentrada = np.fft.ifftshift(transformada_filtrada)
    imagen_compleja = np.fft.ifft2(transformada_descentrada)
    return np.real(imagen_compleja)


def aplicar_pasa_alto_frecuencia_acentuado(imagen, radio, tipo_filtro="ideal"):
    from src.app.suavizado_frecuencia import (
        calcular_fft,
        calcular_espectro_fft,
        visualizar_mascara,
        aplicar_filtro_frecuencia,
    )

    base = convertir_grises(imagen) if imagen.ndim == 3 else imagen
    if tipo_filtro == "gaussiano":
        mascara = crear_mascara_pasa_alto_gaussiana(base.shape, radio)
    else:
        mascara = crear_mascara_pasa_alto(base.shape, radio)
    transformada = calcular_fft(base)
    espectro_original = calcular_espectro_fft(transformada)
    transformada_filtrada = aplicar_filtro_frecuencia(transformada, mascara)
    espectro_filtrado = calcular_espectro_fft(transformada_filtrada)
    respuesta = reconstruir_respuesta_ifft(transformada_filtrada)
    acentuada_gris = np.clip(base.astype(np.float32) + respuesta, 0, 255).astype(
        np.uint8
    )
    if imagen.ndim == 2:
        return (
            espectro_original,
            espectro_filtrado,
            visualizar_mascara(mascara),
            acentuada_gris,
        )

    # Para RGB: aplicar filtro pasa alto a cada canal y sumar al original
    def procesar_canal(canal):
        t = calcular_fft(canal)
        tf = aplicar_filtro_frecuencia(t, mascara)
        r = reconstruir_respuesta_ifft(tf)
        return np.clip(canal.astype(np.float32) + r, 0, 255).astype(np.uint8)

    acentuada = aplicar_por_canal(imagen, procesar_canal)
    return espectro_original, espectro_filtrado, visualizar_mascara(mascara), acentuada


def aplicar_acentuado(imagen, dominio, tipo_acentuacion, radio):

    if dominio == "Frecuencia":
        tipo_filtro = (
            "gaussiano" if tipo_acentuacion == "Pasa altas gaussiano" else "ideal"
        )

        espectro_orig, espectro_filt, mascara, acentuada = (
            aplicar_pasa_alto_frecuencia_acentuado(
                imagen,
                radio,
                tipo_filtro,
            )
        )

        return acentuada, espectro_orig, espectro_filt, mascara
    # Dominio espacial
    if tipo_acentuacion == "Laplaciano 8 vecinos":
        acentuada = aplicar_por_canal(imagen, laplaciano_8_vecinos)
    else:
        acentuada = aplicar_por_canal(imagen, laplaciano_espacial)

    return acentuada, None, None, None
