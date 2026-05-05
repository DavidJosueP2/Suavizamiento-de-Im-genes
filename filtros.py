import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view


def cargar_imagen(ruta_imagen):
    return np.array(Image.open(ruta_imagen).convert("RGB"), dtype=np.uint8)


def convertir_grises(imagen_rgb):
    gris = (
        0.299 * imagen_rgb[:, :, 0].astype(np.float32)
        + 0.587 * imagen_rgb[:, :, 1].astype(np.float32)
        + 0.114 * imagen_rgb[:, :, 2].astype(np.float32)
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


def _ventanas_aplanadas(imagen, tamano_mascara):
    padded = aplicar_padding_replicado(imagen, tamano_mascara)
    ventanas = sliding_window_view(padded, (tamano_mascara, tamano_mascara))
    return ventanas.reshape(
        imagen.shape[0], imagen.shape[1], tamano_mascara * tamano_mascara
    )


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
        return aplicar_por_canal(
            imagen, lambda canal: filtro_media(canal, tamano_mascara)
        )
    if tipo_filtro == "Mediana":
        return aplicar_por_canal(
            imagen, lambda canal: filtro_mediana(canal, tamano_mascara)
        )
    return aplicar_por_canal(imagen, lambda canal: filtro_moda(canal, tamano_mascara))


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
    mascara = np.zeros((alto, ancho), dtype=np.float64)

    for fila in range(alto):
        distancia_y = (fila - centro_y) ** 2
        for columna in range(ancho):
            distancia = distancia_y + (columna - centro_x) ** 2
            if distancia <= radio_cuadrado:
                mascara[fila, columna] = 1.0

    return mascara


def aplicar_filtro_frecuencia(transformada_centrada, mascara):
    return transformada_centrada * mascara


def reconstruir_imagen_ifft(transformada_filtrada):
    transformada_descentrada = np.fft.ifftshift(transformada_filtrada)
    imagen_compleja = np.fft.ifft2(transformada_descentrada)
    magnitud = np.abs(imagen_compleja)
    return _normalizar_visual(magnitud)


def _espectro_con_mascara(espectro, mascara):
    salida = espectro.copy()
    salida[mascara == 0] = (salida[mascara == 0] * 0.25).astype(np.uint8)
    return salida


def _aplicar_frecuencia_canal(canal, radio):
    transformada = calcular_fft(canal)
    espectro = calcular_espectro_fft(transformada)
    mascara = crear_mascara_pasa_bajo(canal.shape, radio)
    transformada_filtrada = aplicar_filtro_frecuencia(transformada, mascara)
    resultado = reconstruir_imagen_ifft(transformada_filtrada)
    return espectro, resultado, mascara


def _imagen_base_espectro(imagen):
    if imagen.ndim == 2:
        return imagen
    return convertir_grises(imagen)


def actualizar_preprocesamiento(imagen_rgb, minimo, maximo, umbral):
    gris = convertir_grises(imagen_rgb)
    histograma_original = calcular_histograma(gris)
    normalizada = normalizar_min_max(gris, int(minimo), int(maximo))
    histograma_normalizado = calcular_histograma(normalizada)
    binaria = binarizar_imagen(normalizada, int(umbral))

    return {
        "original": imagen_rgb,
        "gris": gris,
        "histograma_original": histograma_original,
        "normalizada": normalizada,
        "histograma_normalizado": histograma_normalizado,
        "binaria": binaria,
    }


KERNEL_DEFAULT = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
]


def procesar_imagen(
    imagen,
    porcentaje_ruido,
    tipo_filtro,
    tamano_mascara,
    radio_fourier,
    aplicar_preprocesamiento=False,
    minimo_norm=0,
    maximo_norm=255,
    umbral=128,
    progreso=None,
    kernel=KERNEL_DEFAULT,
):
    if progreso:
        progreso(0.05)

    preprocesamiento = actualizar_preprocesamiento(
        imagen, minimo_norm, maximo_norm, umbral
    )
    base_proceso = preprocesamiento["binaria"] if aplicar_preprocesamiento else imagen
    if progreso:
        progreso(0.20)
    kernel_generado = generar_kernel(tamano_mascara)
    imagen_convolucion = aplicar_por_canal(
        base_proceso,
        lambda canal: convolucion_manual(canal, kernel_generado),
    )
    imagen_ruido = agregar_ruido_sal_pimienta(imagen_convolucion, porcentaje_ruido)
    if progreso:
        progreso(0.40)

    resultado_espacial = aplicar_filtro_espacial(
        imagen_ruido, tipo_filtro, tamano_mascara
    )
    if progreso:
        progreso(0.65)

    canal_espectro = _imagen_base_espectro(imagen_ruido)
    espectro, resultado_frecuencia, mascara = _aplicar_frecuencia_canal(
        canal_espectro, radio_fourier
    )
    espectro = _espectro_con_mascara(espectro, mascara)

    if imagen_ruido.ndim == 3:
        resultado_frecuencia = aplicar_por_canal(
            imagen_ruido,
            lambda canal: _aplicar_frecuencia_canal(canal, radio_fourier)[1],
        )

    if progreso:
        progreso(1.0)

    return {
        "preprocesamiento": preprocesamiento,
        "base_proceso": base_proceso,
        "convolucion": imagen_convolucion,  # 👈 NUEVO
        "ruido": imagen_ruido,
        "espacial": resultado_espacial,
        "espectro": espectro,
        "frecuencia": resultado_frecuencia,
    }


def convolucion_manual(imagen, kernel):
    kernel = np.array(kernel, dtype=np.float32)
    k = kernel.shape[0]
    radio = k // 2

    padded = aplicar_padding_replicado(imagen, k)
    salida = np.zeros_like(imagen, dtype=np.float32)

    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            region = padded[i : i + k, j : j + k]
            salida[i, j] = np.sum(region * kernel)

    return _normalizar_visual(salida)


def generar_kernel(tamano):
    k = int(tamano)
    kernel = np.ones((k, k), dtype=np.float32)
    kernel /= k * k
    return kernel


def guardar_resultado(imagen, ruta_salida):
    Image.fromarray(imagen.astype(np.uint8)).save(ruta_salida)
