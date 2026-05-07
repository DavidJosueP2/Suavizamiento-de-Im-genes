import numpy as np
from PIL import Image, ImageDraw
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


def detectar_bounding_box(imagen_binaria):
    regiones = detectar_regiones(imagen_binaria, min_area=100)
    if not regiones:
        return None

    region = max(regiones, key=lambda item: item["area"])
    return (
        int(region["x"]),
        int(region["y"]),
        int(region["x"] + region["width"] - 1),
        int(region["y"] + region["height"] - 1),
    )


def detectar_regiones(imagen_binaria, min_area=100):
    imagen_binaria = imagen_binaria.astype(np.uint8)
    alto, ancho = imagen_binaria.shape
    blancos = imagen_binaria > 0
    negros = ~blancos

    mascara = negros if negros.sum() <= blancos.sum() else blancos
    visitado = np.zeros((alto, ancho), dtype=bool)
    regiones = []

    for fila in range(alto):
        for columna in range(ancho):
            if not mascara[fila, columna] or visitado[fila, columna]:
                continue

            pila = [(fila, columna)]
            visitado[fila, columna] = True
            min_fila = max_fila = fila
            min_columna = max_columna = columna
            area = 0

            while pila:
                actual_fila, actual_columna = pila.pop()
                area += 1
                min_fila = min(min_fila, actual_fila)
                max_fila = max(max_fila, actual_fila)
                min_columna = min(min_columna, actual_columna)
                max_columna = max(max_columna, actual_columna)

                for df in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if df == 0 and dc == 0:
                            continue

                        nf = actual_fila + df
                        nc = actual_columna + dc
                        if nf < 0 or nf >= alto or nc < 0 or nc >= ancho:
                            continue
                        if visitado[nf, nc] or not mascara[nf, nc]:
                            continue

                        visitado[nf, nc] = True
                        pila.append((nf, nc))

            if area < min_area:
                continue

            regiones.append(
                {
                    "area": int(area),
                    "x": int(min_columna),
                    "y": int(min_fila),
                    "width": int(max_columna - min_columna + 1),
                    "height": int(max_fila - min_fila + 1),
                    "centroid_x": float((min_columna + max_columna) / 2),
                    "centroid_y": float((min_fila + max_fila) / 2),
                }
            )

    regiones.sort(key=lambda item: item["area"], reverse=True)
    for indice, region in enumerate(regiones, start=1):
        region["label"] = indice
    return regiones


def dibujar_bounding_box(imagen, bounding_box):
    if imagen.ndim == 2:
        salida = np.stack([imagen, imagen, imagen], axis=2).astype(np.uint8)
    else:
        salida = imagen.copy().astype(np.uint8)

    if bounding_box is None:
        return salida

    x1, y1, x2, y2 = bounding_box
    alto, ancho = salida.shape[:2]
    x1 = max(0, min(ancho - 1, int(x1)))
    x2 = max(0, min(ancho - 1, int(x2)))
    y1 = max(0, min(alto - 1, int(y1)))
    y2 = max(0, min(alto - 1, int(y2)))
    color = np.array([255, 0, 0], dtype=np.uint8)

    for grosor in range(3):
        yy1 = max(0, y1 - grosor)
        yy2 = min(alto - 1, y2 + grosor)
        xx1 = max(0, x1 - grosor)
        xx2 = min(ancho - 1, x2 + grosor)

        salida[yy1, xx1 : xx2 + 1] = color
        salida[yy2, xx1 : xx2 + 1] = color
        salida[yy1 : yy2 + 1, xx1] = color
        salida[yy1 : yy2 + 1, xx2] = color

    return salida


def dibujar_regiones_numeradas(imagen, regiones):
    base = dibujar_bounding_box(imagen, None)
    pil = Image.fromarray(base)
    dibujo = ImageDraw.Draw(pil)

    for region in regiones:
        x = int(region["x"])
        y = int(region["y"])
        ancho = int(region["width"])
        alto = int(region["height"])
        x2 = x + ancho - 1
        y2 = y + alto - 1
        etiqueta = str(region["label"])

        for offset in range(3):
            dibujo.rectangle((x - offset, y - offset, x2 + offset, y2 + offset), outline=(0, 180, 70))

        texto_x = x
        texto_y = max(0, y - 22)
        dibujo.rectangle((texto_x, texto_y, texto_x + 26, texto_y + 18), fill=(255, 255, 255))
        dibujo.text((texto_x + 6, texto_y + 2), etiqueta, fill=(0, 90, 180))

    return np.array(pil, dtype=np.uint8)


def recortar_regiones(imagen, regiones, margen=8):
    if imagen.ndim == 2:
        base = np.stack([imagen, imagen, imagen], axis=2).astype(np.uint8)
    else:
        base = imagen.astype(np.uint8)

    alto, ancho = base.shape[:2]
    recortes = []
    for region in regiones:
        x1 = max(0, int(region["x"]) - margen)
        y1 = max(0, int(region["y"]) - margen)
        x2 = min(ancho, int(region["x"] + region["width"]) + margen)
        y2 = min(alto, int(region["y"] + region["height"]) + margen)
        recorte = base[y1:y2, x1:x2].copy()
        recortes.append({"region": region, "imagen": recorte})

    return recortes


def actualizar_preprocesamiento(imagen_rgb, minimo, maximo, umbral):
    gris = convertir_grises(imagen_rgb)

    return {
        "original": imagen_rgb,
        "gris": gris,
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


def prewitt_horizontal(imagen):
    kernel = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    return convolucion_manual(imagen, kernel)


def prewitt_vertical(imagen):
    kernel = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    return convolucion_manual(imagen, kernel)


def prewitt_combinado(imagen):
    gx = prewitt_vertical(imagen).astype(np.float32)
    gy = prewitt_horizontal(imagen).astype(np.float32)
    return _normalizar_visual(np.sqrt(gx * gx + gy * gy))


def laplaciano_espacial(imagen):
    kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    respuesta = convolucion_cruda(imagen, kernel)
    realzada = imagen.astype(np.float32) + respuesta
    return np.clip(realzada, 0, 255).astype(np.uint8)


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
    return aplicar_por_canal(imagen, lambda canal: filtro_mediana(canal, tamano_mascara))


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


def visualizar_mascara(mascara):
    return (mascara * 255).astype(np.uint8)


def aplicar_filtro_frecuencia(transformada_centrada, mascara):
    return transformada_centrada * mascara


def reconstruir_imagen_ifft(transformada_filtrada):
    transformada_descentrada = np.fft.ifftshift(transformada_filtrada)
    imagen_compleja = np.fft.ifft2(transformada_descentrada)
    magnitud = np.abs(imagen_compleja)
    return _normalizar_visual(magnitud)


def reconstruir_respuesta_ifft(transformada_filtrada):
    transformada_descentrada = np.fft.ifftshift(transformada_filtrada)
    imagen_compleja = np.fft.ifft2(transformada_descentrada)
    return np.real(imagen_compleja)


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
    return espectro_original, espectro_filtrado, visualizar_mascara(mascara), reconstruida


def aplicar_filtros_frecuencia_dobles(imagen, radio):
    base_espectro = convertir_grises(imagen) if imagen.ndim == 3 else imagen
    espectro_original, espectro_bajo, mascara_baja, frecuencia_baja = _aplicar_frecuencia_canal(
        base_espectro, radio, "Pasa bajas"
    )
    _, espectro_alto, mascara_alta, frecuencia_alta = _aplicar_frecuencia_canal(
        base_espectro, radio, "Pasa altas"
    )

    if imagen.ndim == 3:
        frecuencia_baja = aplicar_por_canal(
            imagen, lambda canal: _aplicar_frecuencia_canal(canal, radio, "Pasa bajas")[3]
        )
        frecuencia_alta = aplicar_por_canal(
            imagen, lambda canal: _aplicar_frecuencia_canal(canal, radio, "Pasa altas")[3]
        )

    return espectro_original, espectro_bajo, espectro_alto, mascara_baja, mascara_alta, frecuencia_baja, frecuencia_alta


def aplicar_filtro_frecuencia_completo(imagen, radio, tipo_frecuencia):
    base_espectro = convertir_grises(imagen) if imagen.ndim == 3 else imagen
    espectro_original, espectro_filtrado, mascara, reconstruida_gris = _aplicar_frecuencia_canal(
        base_espectro, radio, tipo_frecuencia
    )

    if imagen.ndim == 2:
        reconstruida = reconstruida_gris
    else:
        reconstruida = aplicar_por_canal(
            imagen, lambda canal: _aplicar_frecuencia_canal(canal, radio, tipo_frecuencia)[3]
        )

    return espectro_original, espectro_filtrado, mascara, reconstruida


def aplicar_suavizado(imagen, dominio, tipo_suavizado, tamano_mascara, radio):
    if dominio == "Frecuencia":
        espectro, espectro_filtrado, mascara, suavizada = aplicar_filtro_frecuencia_completo(
            imagen, radio, "Pasa bajas"
        )
        return suavizada, espectro, espectro_filtrado, mascara

    suavizada = aplicar_filtro_espacial(imagen, tipo_suavizado, tamano_mascara)
    return suavizada, None, None, None


def aplicar_acentuacion_bordes(imagen, tipo_acentuacion):
    return aplicar_por_canal(imagen, laplaciano_espacial)


def aplicar_acentuado(imagen, dominio, tipo_acentuacion, radio):
    if dominio == "Frecuencia":
        espectro, espectro_filtrado, mascara, acentuada = aplicar_pasa_alto_frecuencia_acentuado(
            imagen, radio
        )
        return acentuada, espectro, espectro_filtrado, mascara

    acentuada = aplicar_acentuacion_bordes(imagen, tipo_acentuacion)
    return acentuada, None, None, None


def _pasa_alto_acentuado_canal(canal, radio):
    transformada = calcular_fft(canal)
    espectro = calcular_espectro_fft(transformada)
    mascara = crear_mascara_pasa_alto(canal.shape, radio)
    transformada_filtrada = aplicar_filtro_frecuencia(transformada, mascara)
    espectro_filtrado = calcular_espectro_fft(transformada_filtrada)
    respuesta = reconstruir_respuesta_ifft(transformada_filtrada)
    acentuada = canal.astype(np.float32) + respuesta
    return espectro, espectro_filtrado, visualizar_mascara(mascara), np.clip(acentuada, 0, 255).astype(np.uint8)


def aplicar_pasa_alto_frecuencia_acentuado(imagen, radio):
    base = convertir_grises(imagen) if imagen.ndim == 3 else imagen
    espectro, espectro_filtrado, mascara, acentuada_gris = _pasa_alto_acentuado_canal(base, radio)

    if imagen.ndim == 2:
        return espectro, espectro_filtrado, mascara, acentuada_gris

    acentuada = aplicar_por_canal(
        imagen, lambda canal: _pasa_alto_acentuado_canal(canal, radio)[3]
    )
    return espectro, espectro_filtrado, mascara, acentuada


def reconocer_bordes_gradiente(imagen, metodo):
    gris = convertir_grises(imagen)
    if metodo == "Sobel":
        gx = convolucion_cruda(gris, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        gy = convolucion_cruda(gris, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    else:
        gx = convolucion_cruda(gris, [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        gy = convolucion_cruda(gris, [[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    magnitud = np.sqrt(gx * gx + gy * gy)
    return _normalizar_visual(gx), _normalizar_visual(gy), _normalizar_visual(magnitud)


def generar_kernel(tamano):
    k = int(tamano)
    kernel = np.ones((k, k), dtype=np.float32)
    kernel /= k * k
    return kernel


def procesar_imagen(
    imagen,
    porcentaje_ruido,
    tipo_suavizado,
    tamano_mascara,
    radio_suavizado,
    dominio_suavizado="Espacial",
    dominio_acentuacion="Espacial",
    tipo_acentuacion="Laplaciano",
    radio_acentuacion=None,
    metodo_gradiente="Prewitt",
    aplicar_preprocesamiento=False,
    minimo_norm=0,
    maximo_norm=255,
    umbral_binario=60,
    min_area_region=10,
    progreso=None,
):
    if progreso:
        progreso(0.05)

    preprocesamiento = actualizar_preprocesamiento(
        imagen, minimo_norm, maximo_norm, umbral_binario
    )
    base_proceso = preprocesamiento["gris"]
    if progreso:
        progreso(0.20)

    ruido = agregar_ruido_sal_pimienta(base_proceso, porcentaje_ruido)
    normalizada_flujo = normalizar_min_max(ruido, minimo_norm, maximo_norm)
    if progreso:
        progreso(0.35)

    suavizado, espectro_suavizado, espectro_suavizado_filtrado, mascara_suavizado = aplicar_suavizado(
        normalizada_flujo, dominio_suavizado, tipo_suavizado, tamano_mascara, radio_suavizado
    )
    if progreso:
        progreso(0.55)

    if radio_acentuacion is None:
        radio_acentuacion = radio_suavizado

    acentuacion, espectro_acentuacion, espectro_acentuacion_filtrado, mascara_acentuacion = aplicar_acentuado(
        suavizado, dominio_acentuacion, tipo_acentuacion, radio_acentuacion
    )
    if progreso:
        progreso(0.75)

    gradiente_x, gradiente_y, gradiente_magnitud = reconocer_bordes_gradiente(
        acentuacion, metodo_gradiente
    )
    final_binaria = binarizar_imagen(gradiente_magnitud, umbral_binario)
    regiones = detectar_regiones(final_binaria, min_area=min_area_region)
    bounding_box = detectar_bounding_box(final_binaria)
    imagen_bounding_box = dibujar_bounding_box(gradiente_magnitud, bounding_box)
    regiones_numeradas = dibujar_regiones_numeradas(gradiente_magnitud, regiones)
    recortes_regiones = recortar_regiones(gradiente_magnitud, regiones)
    if progreso:
        progreso(1.0)

    return {
        "preprocesamiento": preprocesamiento,
        "base_proceso": base_proceso,
        "ruido": ruido,
        "normalizada_flujo": normalizada_flujo,
        "suavizado": suavizado,
        "espectro_suavizado": espectro_suavizado,
        "espectro_suavizado_filtrado": espectro_suavizado_filtrado,
        "mascara_suavizado": mascara_suavizado,
        "acentuacion": acentuacion,
        "espectro_acentuacion": espectro_acentuacion,
        "espectro_acentuacion_filtrado": espectro_acentuacion_filtrado,
        "mascara_acentuacion": mascara_acentuacion,
        "gradiente_x": gradiente_x,
        "gradiente_y": gradiente_y,
        "gradiente_magnitud": gradiente_magnitud,
        "final_binaria": final_binaria,
        "regiones": regiones,
        "bounding_box": bounding_box,
        "imagen_bounding_box": imagen_bounding_box,
        "regiones_numeradas": regiones_numeradas,
        "recortes_regiones": recortes_regiones,
    }


def guardar_resultado(imagen, ruta_salida):
    Image.fromarray(imagen.astype(np.uint8)).save(ruta_salida)
