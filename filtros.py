import numpy as np
from PIL import Image


def cargar_imagen(ruta_imagen):
    return Image.open(ruta_imagen).convert("RGB")


def convertir_grises(imagen):
    return np.array(imagen.convert("L"), dtype=np.uint8)


def agregar_ruido_sal_pimienta(imagen_gris, porcentaje_ruido):
    porcentaje = max(0.0, min(float(porcentaje_ruido), 100.0)) / 100.0
    salida = imagen_gris.copy()
    
    if porcentaje == 0:
        return salida

    aleatorios = np.random.random(imagen_gris.shape)
    sal = aleatorios < (porcentaje / 2.0)
    pimienta = (aleatorios >= (porcentaje / 2.0)) & (aleatorios < porcentaje)

    salida[sal] = 255
    salida[pimienta] = 0
    return salida.astype(np.uint8)


def aplicar_padding_replicado(imagen_gris):
    alto, ancho = imagen_gris.shape
    padded = np.zeros((alto + 2, ancho + 2), dtype=np.uint8)

    padded[1:alto + 1, 1:ancho + 1] = imagen_gris

    for columna in range(ancho):
        padded[0, columna + 1] = imagen_gris[0, columna]
        padded[alto + 1, columna + 1] = imagen_gris[alto - 1, columna]

    for fila in range(alto):
        padded[fila + 1, 0] = imagen_gris[fila, 0]
        padded[fila + 1, ancho + 1] = imagen_gris[fila, ancho - 1]

    padded[0, 0] = imagen_gris[0, 0]
    padded[0, ancho + 1] = imagen_gris[0, ancho - 1]
    padded[alto + 1, 0] = imagen_gris[alto - 1, 0]
    padded[alto + 1, ancho + 1] = imagen_gris[alto - 1, ancho - 1]
    
    return padded


def filtro_media_3x3(imagen_gris):
    alto, ancho = imagen_gris.shape
    padded = aplicar_padding_replicado(imagen_gris)
    salida = np.zeros_like(imagen_gris, dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            suma = 0
            for i in range(3):
                for j in range(3):
                    suma += int(padded[fila + i, columna + j])
            salida[fila, columna] = int(round(suma / 9.0))

    return salida


def filtro_mediana_3x3(imagen_gris):
    alto, ancho = imagen_gris.shape
    padded = aplicar_padding_replicado(imagen_gris)
    salida = np.zeros_like(imagen_gris, dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            valores = []
            for i in range(3):
                for j in range(3):
                    valores.append(int(padded[fila + i, columna + j]))
            valores.sort()
            salida[fila, columna] = valores[4]

    return salida


def filtro_moda_3x3(imagen_gris):
    alto, ancho = imagen_gris.shape
    padded = aplicar_padding_replicado(imagen_gris)
    salida = np.zeros_like(imagen_gris, dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            conteo = {}
            for i in range(3):
                for j in range(3):
                    valor = int(padded[fila + i, columna + j])
                    conteo[valor] = conteo.get(valor, 0) + 1

            max_repeticiones = max(conteo.values())
            candidatos = [valor for valor, repeticiones in conteo.items() if repeticiones == max_repeticiones]
            pixel_central = int(imagen_gris[fila, columna])
            salida[fila, columna] = min(candidatos, key=lambda valor: (abs(valor - pixel_central), valor))

    return salida


def calcular_espectro_fourier(imagen_gris):
    return np.fft.fftshift(np.fft.fft2(imagen_gris))


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


def procesar_imagen(imagen_gris, porcentaje_ruido, tipo_filtro, radio_fourier, progreso=None):
    if progreso:
        progreso(0.05)

    imagen_ruido = agregar_ruido_sal_pimienta(imagen_gris, porcentaje_ruido)
    if progreso:
        progreso(0.25)

    if tipo_filtro == "Media":
        resultado_espacial = filtro_media_3x3(imagen_ruido)
    elif tipo_filtro == "Mediana":
        resultado_espacial = filtro_mediana_3x3(imagen_ruido)
    else:
        resultado_espacial = filtro_moda_3x3(imagen_ruido)
    if progreso:
        progreso(0.55)

    transformada_centrada = calcular_espectro_fourier(imagen_ruido)
    if progreso:
        progreso(0.70)

    transformada_filtrada = aplicar_mascara_circular_fourier(transformada_centrada, radio_fourier)
    if progreso:
        progreso(0.85)

    resultado_frecuencia = reconstruir_imagen_fourier(transformada_filtrada)
    if progreso:
        progreso(1.0)

    return imagen_ruido, resultado_espacial, resultado_frecuencia
