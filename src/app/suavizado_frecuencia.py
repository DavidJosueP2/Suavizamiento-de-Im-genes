import numpy as np
from src.app.utils import (
    normalizar_visual,
    aplicar_por_canal,
    convertir_grises,
)


# Filtros en frecuencia
def calcular_fft(imagen):
    return np.fft.fftshift(np.fft.fft2(imagen))


def calcular_espectro_fft(transformada_centrada):
    magnitud = np.log1p(np.abs(transformada_centrada))
    return normalizar_visual(magnitud)


# Pasa bajo ideal
def crear_mascara_pasa_bajo(forma, radio):
    alto, ancho = forma[:2]
    cy, cx = alto // 2, ancho // 2
    radio2 = int(radio) ** 2
    y, x = np.ogrid[:alto, :ancho]
    distancia2 = (y - cy) ** 2 + (x - cx) ** 2
    return (distancia2 <= radio2).astype(np.float64)


# Pasa bajo gausiano
def crear_mascara_pasa_bajo_gaussiana(forma, radio):
    alto, ancho = forma[:2]
    cy = alto // 2
    cx = ancho // 2
    y, x = np.ogrid[:alto, :ancho]
    distancia2 = (y - cy) ** 2 + (x - cx) ** 2
    radio2 = float(radio) ** 2
    mascara = np.exp(-distancia2 / (2 * radio2))
    return mascara.astype(np.float64)


def visualizar_mascara(mascara):
    return (mascara * 255).astype(np.uint8)


def aplicar_filtro_frecuencia(transformada_centrada, mascara):
    return transformada_centrada * mascara


def reconstruir_imagen_ifft(transformada_filtrada):
    transformada_descentrada = np.fft.ifftshift(transformada_filtrada)
    imagen_compleja = np.fft.ifft2(transformada_descentrada)
    magnitud = np.abs(imagen_compleja)
    return normalizar_visual(magnitud)


def aplicar_filtro_frecuencia_completo(imagen, radio, tipo_pasa):
    # tipo_pasa: "bajas" o "altas"
    from src.app.acentuado import (
        crear_mascara_pasa_alto,
    )  # import local para evitar circularidad

    base = convertir_grises(imagen) if imagen.ndim == 3 else imagen
    transformada = calcular_fft(base)
    espectro_original = calcular_espectro_fft(transformada)
    if tipo_pasa == "bajas":
        mascara = crear_mascara_pasa_bajo(base.shape, radio)

    elif tipo_pasa == "bajas_gaussiano":
        mascara = crear_mascara_pasa_bajo_gaussiana(base.shape, radio)

    else:
        mascara = crear_mascara_pasa_alto(base.shape, radio)
    transformada_filtrada = aplicar_filtro_frecuencia(transformada, mascara)
    espectro_filtrado = calcular_espectro_fft(transformada_filtrada)
    reconstruida_gris = reconstruir_imagen_ifft(transformada_filtrada)
    if imagen.ndim == 2:
        return (
            espectro_original,
            espectro_filtrado,
            visualizar_mascara(mascara),
            reconstruida_gris,
        )

    # Para RGB, aplicar a cada canal
    def procesar_canal(canal):
        t = calcular_fft(canal)
        tf = aplicar_filtro_frecuencia(t, mascara)
        return reconstruir_imagen_ifft(tf)

    reconstruida = aplicar_por_canal(imagen, procesar_canal)
    return (
        espectro_original,
        espectro_filtrado,
        visualizar_mascara(mascara),
        reconstruida,
    )
