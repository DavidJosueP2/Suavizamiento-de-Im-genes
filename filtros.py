import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view


def cargar_imagen(ruta_imagen):
    return np.array(Image.open(ruta_imagen).convert("RGB"), dtype=np.uint8)


def convertir_grises(imagen):
    if imagen.ndim == 2:
        return imagen.astype(np.uint8)

    gris = (
        0.299 * imagen[:, :, 0].astype(np.float32)
        + 0.587 * imagen[:, :, 1].astype(np.float32)
        + 0.114 * imagen[:, :, 2].astype(np.float32)
    )
    return np.clip(np.rint(gris), 0, 255).astype(np.uint8)


def calcular_histograma(imagen):
    return np.bincount(imagen.ravel(), minlength=256).astype(np.int64)


def normalizar_min_max(imagen, minimo=None, maximo=None):
    if minimo is None:
        minimo = int(imagen.min())
    if maximo is None:
        maximo = int(imagen.max())

    if maximo <= minimo:
        return np.zeros_like(imagen, dtype=np.uint8)

    normalizada = ((imagen.astype(np.float32) - minimo) / (maximo - minimo)) * 255.0
    return np.clip(np.rint(normalizada), 0, 255).astype(np.uint8)


def binarizar_imagen(imagen, umbral):
    return np.where(imagen >= int(umbral), 255, 0).astype(np.uint8)


def actualizar_preprocesamiento(imagen_rgb, minimo, maximo, umbral):
    gris = convertir_grises(imagen_rgb)
    normalizada = normalizar_min_max(gris, int(minimo), int(maximo))
    binaria = binarizar_imagen(normalizada, int(umbral))

    return {
        "original": imagen_rgb,
        "gris": gris,
        "histograma_original": calcular_histograma(gris),
        "normalizada": normalizada,
        "histograma_normalizado": calcular_histograma(normalizada),
        "binaria": binaria,
    }


def agregar_ruido_sal_pimienta(imagen, porcentaje_ruido):
    porcentaje = max(0.0, min(float(porcentaje_ruido), 100.0)) / 100.0
    salida = imagen.copy()

    if porcentaje == 0:
        return salida.astype(np.uint8)

    alto, ancho = imagen.shape[:2]
    aleatorios = np.random.random((alto, ancho))
    sal = aleatorios < (porcentaje / 2.0)
    pimienta = (aleatorios >= (porcentaje / 2.0)) & (aleatorios < porcentaje)

    salida[sal] = 255
    salida[pimienta] = 0
    return salida.astype(np.uint8)


def aplicar_padding_replicado(imagen, tamano_mascara):
    radio = tamano_mascara // 2
    alto, ancho = imagen.shape
    padded = np.zeros((alto + 2 * radio, ancho + 2 * radio), dtype=np.uint8)

    padded[radio : radio + alto, radio : radio + ancho] = imagen
    padded[:radio, radio : radio + ancho] = imagen[0:1, :]
    padded[radio + alto :, radio : radio + ancho] = imagen[alto - 1 : alto, :]
    padded[:, :radio] = padded[:, radio : radio + 1]
    padded[:, radio + ancho :] = padded[:, radio + ancho - 1 : radio + ancho]
    return padded


def _ventanas_aplanadas(imagen, tamano_mascara):
    padded = aplicar_padding_replicado(imagen, tamano_mascara)
    ventanas = sliding_window_view(padded, (tamano_mascara, tamano_mascara))
    return ventanas.reshape(imagen.shape[0], imagen.shape[1], tamano_mascara * tamano_mascara)


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
    ventanas = _ventanas_aplanadas(imagen, tamano_mascara)
    return np.partition(ventanas, centro, axis=2)[:, :, centro].astype(np.uint8)


def filtro_moda(imagen, tamano_mascara):
    total = tamano_mascara * tamano_mascara
    valores = np.sort(_ventanas_aplanadas(imagen, tamano_mascara), axis=2)
    centro = imagen.astype(np.int16)

    mejor_valor = np.zeros_like(imagen, dtype=np.uint8)
    mejor_repeticiones = np.zeros(imagen.shape, dtype=np.uint16)
    mejor_distancia = np.full(imagen.shape, 256, dtype=np.int16)
    valor_actual = valores[:, :, 0].copy()
    repeticiones = np.ones(imagen.shape, dtype=np.uint16)

    for indice in range(1, total):
        mismo_valor = valores[:, :, indice] == valor_actual
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
            valor_actual[cambio] = valores[:, :, indice][cambio]
            repeticiones[cambio] = 1

        repeticiones[mismo_valor] += 1

    distancia = np.abs(valor_actual.astype(np.int16) - centro)
    reemplazar = (repeticiones > mejor_repeticiones) | (
        (repeticiones == mejor_repeticiones) & (distancia < mejor_distancia)
    )
    mejor_valor[reemplazar] = valor_actual[reemplazar]
    return mejor_valor


def convolucion_manual(imagen, kernel):
    return _normalizar_visual(convolucion_cruda(imagen, kernel))


def convolucion_cruda(imagen, kernel):
    kernel = np.array(kernel, dtype=np.float32)
    k = kernel.shape[0]
    ventanas = sliding_window_view(aplicar_padding_replicado(imagen, k), (k, k))
    return np.sum(ventanas * kernel, axis=(2, 3))


def sobel_horizontal(imagen):
    kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    return convolucion_manual(imagen, kernel)


def sobel_vertical(imagen):
    kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    return convolucion_manual(imagen, kernel)


def sobel_combinado(imagen):
    gx = sobel_vertical(imagen).astype(np.float32)
    gy = sobel_horizontal(imagen).astype(np.float32)
    return _normalizar_visual(np.sqrt(gx * gx + gy * gy))


def high_pass_espacial(imagen):
    kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    return convolucion_manual(imagen, kernel)


def aplicar_por_canal(imagen, funcion):
    if imagen.ndim == 2:
        return funcion(imagen)

    canales = []
    for canal in range(imagen.shape[2]):
        canales.append(funcion(imagen[:, :, canal]))
    return np.stack(canales, axis=2).astype(np.uint8)


def aplicar_filtro_espacial(imagen, tipo_filtro, tamano_mascara):
    tamano_mascara = int(tamano_mascara)
    if tamano_mascara % 2 == 0:
        raise ValueError("El tamano de mascara debe ser impar.")

    if tipo_filtro == "Media":
        return aplicar_por_canal(imagen, lambda canal: filtro_media(canal, tamano_mascara))
    if tipo_filtro == "Mediana":
        return aplicar_por_canal(imagen, lambda canal: filtro_mediana(canal, tamano_mascara))
    if tipo_filtro == "Moda":
        return aplicar_por_canal(imagen, lambda canal: filtro_moda(canal, tamano_mascara))
    if tipo_filtro == "High-pass":
        return aplicar_por_canal(imagen, high_pass_espacial)
    if tipo_filtro == "Sobel horizontal":
        return aplicar_por_canal(imagen, sobel_horizontal)
    if tipo_filtro == "Sobel vertical":
        return aplicar_por_canal(imagen, sobel_vertical)
    return aplicar_por_canal(imagen, sobel_combinado)


def calcular_fft(imagen):
    return np.fft.fftshift(np.fft.fft2(imagen))


def _normalizar_visual(imagen):
    minimo = imagen.min()
    maximo = imagen.max()
    if maximo == minimo:
        return np.zeros_like(imagen, dtype=np.uint8)
    normalizada = (imagen - minimo) / (maximo - minimo) * 255.0
    return np.clip(normalizada, 0, 255).astype(np.uint8)


def calcular_espectro_fft(transformada_centrada):
    magnitud = np.log1p(np.abs(transformada_centrada))
    return _normalizar_visual(magnitud)


def crear_mascara_pasa_bajo(forma, radio):
    alto, ancho = forma[:2]
    centro_y = alto // 2
    centro_x = ancho // 2
    radio_cuadrado = int(radio) ** 2
    y, x = np.ogrid[:alto, :ancho]
    distancia = (y - centro_y) ** 2 + (x - centro_x) ** 2
    return (distancia <= radio_cuadrado).astype(np.float64)


def crear_mascara_pasa_alto(forma, radio):
    return 1.0 - crear_mascara_pasa_bajo(forma, radio)


def aplicar_filtro_frecuencia(transformada_centrada, mascara):
    return transformada_centrada * mascara


def reconstruir_imagen_ifft(transformada_filtrada):
    transformada_descentrada = np.fft.ifftshift(transformada_filtrada)
    imagen_compleja = np.fft.ifft2(transformada_descentrada)
    magnitud = np.abs(imagen_compleja)
    return _normalizar_visual(magnitud)


def _aplicar_frecuencia_canal(canal, radio, tipo_frecuencia):
    transformada = calcular_fft(canal)
    espectro_original = calcular_espectro_fft(transformada)

    if tipo_frecuencia == "Pasa altas":
        mascara = crear_mascara_pasa_alto(canal.shape, radio)
    else:
        mascara = crear_mascara_pasa_bajo(canal.shape, radio)

    transformada_filtrada = aplicar_filtro_frecuencia(transformada, mascara)
    espectro_filtrado = calcular_espectro_fft(transformada_filtrada)
    reconstruida = reconstruir_imagen_ifft(transformada_filtrada)
    return espectro_original, espectro_filtrado, reconstruida


def aplicar_filtro_frecuencia_completo(imagen, radio, tipo_frecuencia):
    base_espectro = convertir_grises(imagen) if imagen.ndim == 3 else imagen
    espectro_original, espectro_filtrado, reconstruida_gris = _aplicar_frecuencia_canal(
        base_espectro, radio, tipo_frecuencia
    )

    if imagen.ndim == 2:
        reconstruida = reconstruida_gris
    else:
        reconstruida = aplicar_por_canal(
            imagen,
            lambda canal: _aplicar_frecuencia_canal(canal, radio, tipo_frecuencia)[2],
        )

    return espectro_original, espectro_filtrado, reconstruida


def supresion_no_maximos(magnitud, angulo):
    alto, ancho = magnitud.shape
    salida = np.zeros((alto, ancho), dtype=np.float32)
    angulo = angulo % 180

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            direccion = angulo[fila, columna]

            if (0 <= direccion < 22.5) or (157.5 <= direccion <= 180):
                vecino1 = magnitud[fila, columna - 1]
                vecino2 = magnitud[fila, columna + 1]
            elif 22.5 <= direccion < 67.5:
                vecino1 = magnitud[fila - 1, columna + 1]
                vecino2 = magnitud[fila + 1, columna - 1]
            elif 67.5 <= direccion < 112.5:
                vecino1 = magnitud[fila - 1, columna]
                vecino2 = magnitud[fila + 1, columna]
            else:
                vecino1 = magnitud[fila - 1, columna - 1]
                vecino2 = magnitud[fila + 1, columna + 1]

            if magnitud[fila, columna] >= vecino1 and magnitud[fila, columna] >= vecino2:
                salida[fila, columna] = magnitud[fila, columna]

    return salida


def histeresis(imagen, umbral_bajo, umbral_alto):
    fuertes = imagen >= umbral_alto
    debiles = (imagen >= umbral_bajo) & (imagen < umbral_alto)
    cambio = True

    while cambio:
        vecinos_fuertes = np.zeros_like(fuertes, dtype=bool)
        vecinos_fuertes[1:, :] |= fuertes[:-1, :]
        vecinos_fuertes[:-1, :] |= fuertes[1:, :]
        vecinos_fuertes[:, 1:] |= fuertes[:, :-1]
        vecinos_fuertes[:, :-1] |= fuertes[:, 1:]
        vecinos_fuertes[1:, 1:] |= fuertes[:-1, :-1]
        vecinos_fuertes[1:, :-1] |= fuertes[:-1, 1:]
        vecinos_fuertes[:-1, 1:] |= fuertes[1:, :-1]
        vecinos_fuertes[:-1, :-1] |= fuertes[1:, 1:]

        nuevos_fuertes = debiles & vecinos_fuertes
        cambio = bool(nuevos_fuertes.any())
        fuertes |= nuevos_fuertes
        debiles &= ~nuevos_fuertes

    return np.where(fuertes, 255, 0).astype(np.uint8)


def canny_histeresis(imagen, umbral_bajo, umbral_alto):
    gris = convertir_grises(imagen)
    suavizada = filtro_media(gris, 3)
    gx = convolucion_cruda(suavizada, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = convolucion_cruda(suavizada, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    magnitud = np.sqrt(gx * gx + gy * gy)
    angulo = np.degrees(np.arctan2(gy, gx))
    no_maximos = supresion_no_maximos(_normalizar_visual(magnitud).astype(np.float32), angulo)
    bordes = histeresis(no_maximos, int(umbral_bajo), int(umbral_alto))
    return _normalizar_visual(magnitud), no_maximos.astype(np.uint8), bordes


def procesar_imagen(
    imagen,
    porcentaje_ruido,
    tipo_filtro,
    tamano_mascara,
    radio_fourier,
    tipo_frecuencia="Pasa bajas",
    aplicar_preprocesamiento=False,
    minimo_norm=0,
    maximo_norm=255,
    umbral_binario=128,
    canny_bajo=50,
    canny_alto=120,
    progreso=None,
):
    if progreso:
        progreso(0.05)

    preprocesamiento = actualizar_preprocesamiento(
        imagen, minimo_norm, maximo_norm, umbral_binario
    )
    base_proceso = preprocesamiento["normalizada"] if aplicar_preprocesamiento else imagen
    if progreso:
        progreso(0.20)

    ruido = agregar_ruido_sal_pimienta(base_proceso, porcentaje_ruido)
    if progreso:
        progreso(0.35)

    espacial = aplicar_filtro_espacial(ruido, tipo_filtro, tamano_mascara)
    if progreso:
        progreso(0.55)

    espectro_original, espectro_filtrado, frecuencia = aplicar_filtro_frecuencia_completo(
        espacial, radio_fourier, tipo_frecuencia
    )
    if progreso:
        progreso(0.75)

    canny_magnitud, canny_no_maximos, bordes = canny_histeresis(
        frecuencia, canny_bajo, canny_alto
    )
    final_binaria = binarizar_imagen(bordes, 128)
    if progreso:
        progreso(1.0)

    return {
        "preprocesamiento": preprocesamiento,
        "base_proceso": base_proceso,
        "ruido": ruido,
        "espacial": espacial,
        "espectro_original": espectro_original,
        "espectro_filtrado": espectro_filtrado,
        "frecuencia": frecuencia,
        "canny_magnitud": canny_magnitud,
        "canny_no_maximos": canny_no_maximos,
        "bordes": bordes,
        "final_binaria": final_binaria,
    }


def guardar_resultado(imagen, ruta_salida):
    Image.fromarray(imagen.astype(np.uint8)).save(ruta_salida)
