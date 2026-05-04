import numpy as np
from PIL import Image


def cargar_imagen(ruta_imagen):
    return np.array(Image.open(ruta_imagen).convert("RGB"), dtype=np.uint8)


def agregar_ruido_sal_pimienta(imagen, porcentaje_ruido):
    porcentaje = max(0.0, min(float(porcentaje_ruido), 100.0)) / 100.0
    salida = imagen.copy()
    
    if porcentaje == 0:
        return salida

    alto, ancho = imagen.shape[:2]
    aleatorios = np.random.random((alto, ancho))
    mascara_ruido = aleatorios < porcentaje

    if imagen.ndim == 3:
        cantidad_pixeles = int(np.count_nonzero(mascara_ruido))
        salida[mascara_ruido, :] = np.random.choice([0, 255], size=(cantidad_pixeles, imagen.shape[2]))
    else:
        sal = aleatorios < (porcentaje / 2.0)
        pimienta = (aleatorios >= (porcentaje / 2.0)) & mascara_ruido
        salida[sal] = 255
        salida[pimienta] = 0

    return salida.astype(np.uint8)


def aplicar_padding_replicado(imagen):
    alto, ancho = imagen.shape
    padded = np.zeros((alto + 2, ancho + 2), dtype=np.uint8)

    padded[1:alto + 1, 1:ancho + 1] = imagen

    for columna in range(ancho):
        padded[0, columna + 1] = imagen[0, columna]
        padded[alto + 1, columna + 1] = imagen[alto - 1, columna]

    for fila in range(alto):
        padded[fila + 1, 0] = imagen[fila, 0]
        padded[fila + 1, ancho + 1] = imagen[fila, ancho - 1]

    padded[0, 0] = imagen[0, 0]
    padded[0, ancho + 1] = imagen[0, ancho - 1]
    padded[alto + 1, 0] = imagen[alto - 1, 0]
    padded[alto + 1, ancho + 1] = imagen[alto - 1, ancho - 1]
    
    return padded


def filtro_media_3x3(imagen):
    alto, ancho = imagen.shape
    padded = aplicar_padding_replicado(imagen)
    salida = np.zeros_like(imagen, dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            suma = 0
            for i in range(3):
                for j in range(3):
                    suma += int(padded[fila + i, columna + j])
            salida[fila, columna] = int(round(suma / 9.0))

    return salida


def filtro_mediana_3x3(imagen):
    alto, ancho = imagen.shape
    padded = aplicar_padding_replicado(imagen)
    salida = np.zeros_like(imagen, dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            valores = []
            for i in range(3):
                for j in range(3):
                    valores.append(int(padded[fila + i, columna + j]))
            valores.sort()
            salida[fila, columna] = valores[4]

    return salida


def filtro_moda_3x3(imagen):
    alto, ancho = imagen.shape
    padded = aplicar_padding_replicado(imagen)
    salida = np.zeros_like(imagen, dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            conteo = {}
            for i in range(3):
                for j in range(3):
                    valor = int(padded[fila + i, columna + j])
                    conteo[valor] = conteo.get(valor, 0) + 1

            max_repeticiones = max(conteo.values())
            candidatos = [valor for valor, repeticiones in conteo.items() if repeticiones == max_repeticiones]
            pixel_central = int(imagen[fila, columna])
            salida[fila, columna] = min(candidatos, key=lambda valor: (abs(valor - pixel_central), valor))

    return salida


def aplicar_por_canal(imagen, funcion):
    if imagen.ndim == 2:
        return funcion(imagen)

    canales = []
    for canal in range(imagen.shape[2]):
        canales.append(funcion(imagen[:, :, canal]))
    return np.stack(canales, axis=2).astype(np.uint8)


def calcular_espectro_fourier(imagen):
    return np.fft.fftshift(np.fft.fft2(imagen))


def aplicar_mascara_circular_fourier(transformada_centrada, radio):
    alto, ancho = transformada_centrada.shape
    centro_y = alto // 2
    centro_x = ancho // 2
    radio = max(0, int(radio))
    mascara = np.zeros((alto, ancho), dtype=np.float64)

    for fila in range(alto):
        for columna in range(ancho):
            distancia = ((fila - centro_y) ** 2 + (columna - centro_x) ** 2) ** 0.5
            if distancia <= radio:
                mascara[fila, columna] = 1.0

    return transformada_centrada * mascara


def reconstruir_imagen_fourier(transformada_filtrada):
    transformada_descentrada = np.fft.ifftshift(transformada_filtrada)
    imagen_compleja = np.fft.ifft2(transformada_descentrada)
    magnitud = np.abs(imagen_compleja)

    minimo = magnitud.min()
    maximo = magnitud.max()
    if maximo == minimo:
        return np.clip(magnitud, 0, 255).astype(np.uint8)

    normalizada = (magnitud - minimo) / (maximo - minimo) * 255.0
    return np.clip(normalizada, 0, 255).astype(np.uint8)


def filtrar_fourier_pasa_bajas(imagen, radio_fourier):
    transformada_centrada = calcular_espectro_fourier(imagen)
    transformada_filtrada = aplicar_mascara_circular_fourier(transformada_centrada, radio_fourier)
    return reconstruir_imagen_fourier(transformada_filtrada)


def procesar_imagen(imagen, porcentaje_ruido, tipo_filtro, radio_fourier, progreso=None):
    if progreso:
        progreso(0.05)

    imagen_ruido = agregar_ruido_sal_pimienta(imagen, porcentaje_ruido)
    if progreso:
        progreso(0.25)

    if tipo_filtro == "Media":
        resultado_espacial = aplicar_por_canal(imagen_ruido, filtro_media_3x3)
    elif tipo_filtro == "Mediana":
        resultado_espacial = aplicar_por_canal(imagen_ruido, filtro_mediana_3x3)
    else:
        resultado_espacial = aplicar_por_canal(imagen_ruido, filtro_moda_3x3)
    if progreso:
        progreso(0.55)

    if progreso:
        progreso(0.70)

    resultado_frecuencia = aplicar_por_canal(imagen_ruido, lambda canal: filtrar_fourier_pasa_bajas(canal, radio_fourier))
    if progreso:
        progreso(0.85)

    if progreso:
        progreso(1.0)

    return imagen_ruido, resultado_espacial, resultado_frecuencia
