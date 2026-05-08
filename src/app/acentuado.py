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


def crear_mascara_pasa_alto(forma, radio):
    from suavizado import crear_mascara_pasa_bajo

    return 1.0 - crear_mascara_pasa_bajo(forma, radio)


def reconstruir_respuesta_ifft(transformada_filtrada):
    transformada_descentrada = np.fft.ifftshift(transformada_filtrada)
    imagen_compleja = np.fft.ifft2(transformada_descentrada)
    return np.real(imagen_compleja)


def aplicar_pasa_alto_frecuencia_acentuado(imagen, radio):
    from suavizado import (
        calcular_fft,
        calcular_espectro_fft,
        visualizar_mascara,
        aplicar_filtro_frecuencia,
    )

    base = convertir_grises(imagen) if imagen.ndim == 3 else imagen
    transformada = calcular_fft(base)
    espectro_original = calcular_espectro_fft(transformada)
    mascara = crear_mascara_pasa_alto(base.shape, radio)
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
        espectro_orig, espectro_filt, mascara, acentuada = (
            aplicar_pasa_alto_frecuencia_acentuado(imagen, radio)
        )
        return acentuada, espectro_orig, espectro_filt, mascara
    # Dominio espacial: Laplaciano
    acentuada = aplicar_por_canal(imagen, laplaciano_espacial)
    return acentuada, None, None, None
