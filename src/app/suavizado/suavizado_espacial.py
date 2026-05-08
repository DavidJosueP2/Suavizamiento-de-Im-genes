import numpy as np
from src.app.utils.utils import (
    aplicar_por_canal,
    ventanas_aplanadas,
    aplicar_padding_replicado,
)


# Filtros espaciales
def filtro_media(imagen, tamano_mascara):
    padded = aplicar_padding_replicado(imagen, tamano_mascara)
    total = tamano_mascara * tamano_mascara
    integral = padded.astype(np.uint64).cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    suma = (
        integral[tamano_mascara:, tamano_mascara:]
        - integral[:-tamano_mascara, tamano_mascara:]
        - integral[tamano_mascara:, :-tamano_mascara]
        + integral[:-tamano_mascara, :-tamano_mascara]
    )
    return np.rint(suma / total).astype(np.uint8)


def filtro_mediana(imagen, tamano_mascara):
    total = tamano_mascara * tamano_mascara
    centro = total // 2
    ventanas = ventanas_aplanadas(imagen, tamano_mascara)
    return np.partition(ventanas, centro, axis=2)[:, :, centro].astype(np.uint8)


def filtro_moda(imagen, tamano_mascara):
    total = tamano_mascara * tamano_mascara
    valores = np.sort(ventanas_aplanadas(imagen, tamano_mascara), axis=2)
    centro = imagen.astype(np.int16)
    mejor_valor = np.zeros_like(imagen, dtype=np.uint8)
    mejor_repeticiones = np.zeros(imagen.shape, dtype=np.uint16)
    mejor_distancia = np.full(imagen.shape, 256, dtype=np.int16)
    valor_actual = valores[:, :, 0].copy()
    repeticiones = np.ones(imagen.shape, dtype=np.uint16)
    for idx in range(1, total):
        mismo_valor = valores[:, :, idx] == valor_actual
        cambio = ~mismo_valor
        if cambio.any():
            distancia = np.abs(valor_actual.astype(np.int16) - centro)
            reemplazar = cambio & (
                (repeticiones > mejor_repeticiones)
                | ((repeticiones == mejor_repeticiones) & (distancia < mejor_distancia))
            )
            mejor_valor[reemplazar] = valor_actual[reemplazar]
            mejor_repeticiones[reemplazar] = repeticiones[reemplazar]
            mejor_distancia[reemplazar] = distancia[reemplazar]
            valor_actual[cambio] = valores[:, :, idx][cambio]
            repeticiones[cambio] = 1
        repeticiones[mismo_valor] += 1
    distancia = np.abs(valor_actual.astype(np.int16) - centro)
    reemplazar = (repeticiones > mejor_repeticiones) | (
        (repeticiones == mejor_repeticiones) & (distancia < mejor_distancia)
    )
    mejor_valor[reemplazar] = valor_actual[reemplazar]
    return mejor_valor


def aplicar_filtro_espacial(imagen, tipo_filtro, tamano_mascara):
    tamano_mascara = int(tamano_mascara)
    if tamano_mascara % 2 == 0:
        raise ValueError("El tamaño de máscara debe ser impar.")
    if tipo_filtro == "Media":
        return aplicar_por_canal(imagen, lambda c: filtro_media(c, tamano_mascara))
    if tipo_filtro == "Mediana":
        return aplicar_por_canal(imagen, lambda c: filtro_mediana(c, tamano_mascara))
    if tipo_filtro == "Moda":
        return aplicar_por_canal(imagen, lambda c: filtro_moda(c, tamano_mascara))
    return aplicar_por_canal(imagen, lambda c: filtro_mediana(c, tamano_mascara))
