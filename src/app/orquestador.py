import numpy as np
from PIL import Image
from src.app.utils.utils import convertir_grises
from src.app.suavizado.suavizado import aplicar_suavizado
from src.app.acentuado.acentuado import aplicar_acentuado
from src.app.gradiente.gradiente import reconocer_bordes_gradiente_direccional
from src.app.utils.binarizacion import binarizar_imagen
from src.app.regiones import (
    detectar_regiones,
    detectar_bounding_box,
    dibujar_bounding_box,
    dibujar_regiones_numeradas,
    recortar_regiones,
)


def cargar_imagen(ruta):
    return np.array(Image.open(ruta).convert("RGB"), dtype=np.uint8)


def normalizar_min_max(imagen, minimo=None, maximo=None):
    if minimo is None:
        minimo = int(imagen.min())
    if maximo is None:
        maximo = int(imagen.max())
    if maximo <= minimo:
        return np.zeros_like(imagen, dtype=np.uint8)
    norm = ((imagen.astype(np.float32) - minimo) / (maximo - minimo)) * 255.0
    return np.clip(np.rint(norm), 0, 255).astype(np.uint8)


def agregar_ruido_sal_pimienta(imagen, porcentaje):
    p = max(0.0, min(float(porcentaje), 100.0)) / 100.0
    salida = imagen.copy()
    if p == 0:
        return salida.astype(np.uint8)
    h, w = imagen.shape[:2]
    rand = np.random.random((h, w))
    sal = rand < (p / 2.0)
    pimienta = (rand >= (p / 2.0)) & (rand < p)
    salida[sal] = 255
    salida[pimienta] = 0
    return salida.astype(np.uint8)


def actualizar_preprocesamiento(imagen_rgb, minimo, maximo, umbral):
    # umbral no se usa aquí, se mantiene por compatibilidad
    return {"original": imagen_rgb, "gris": convertir_grises(imagen_rgb)}


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

    pre = actualizar_preprocesamiento(imagen, minimo_norm, maximo_norm, umbral_binario)
    base_proceso = pre["gris"]
    if progreso:
        progreso(0.20)

    ruido = agregar_ruido_sal_pimienta(base_proceso, porcentaje_ruido)
    normalizada_flujo = normalizar_min_max(ruido, minimo_norm, maximo_norm)
    if progreso:
        progreso(0.35)

    suavizado, espectro_s, _, mascara_s = aplicar_suavizado(
        normalizada_flujo,
        dominio_suavizado,
        tipo_suavizado,
        tamano_mascara,
        radio_suavizado,
    )
    if progreso:
        progreso(0.55)

    if radio_acentuacion is None:
        radio_acentuacion = radio_suavizado
    acentuacion, espectro_a, _, mascara_a = aplicar_acentuado(
        suavizado, dominio_acentuacion, tipo_acentuacion, radio_acentuacion
    )
    if progreso:
        progreso(0.75)

    gradientes = reconocer_bordes_gradiente_direccional(acentuacion, metodo_gradiente)
    grad_0 = gradientes["0"]
    grad_45 = gradientes["45"]
    grad_90 = gradientes["90"]
    grad_135 = gradientes["135"]
    grad_mag = gradientes["magnitud"]

    final_binaria = binarizar_imagen(grad_mag, umbral_binario)
    regiones = detectar_regiones(final_binaria, min_area=min_area_region)
    bounding_box = detectar_bounding_box(final_binaria, min_area=min_area_region)
    img_bbox = dibujar_bounding_box(grad_mag, bounding_box)
    regiones_numeradas = dibujar_regiones_numeradas(grad_mag, regiones)
    recortes = recortar_regiones(grad_mag, regiones)

    if progreso:
        progreso(1.0)

    return {
        "preprocesamiento": pre,
        "base_proceso": base_proceso,
        "ruido": ruido,
        "normalizada_flujo": normalizada_flujo,
        "suavizado": suavizado,
        "espectro_suavizado": espectro_s,  # puede ser None
        "mascara_suavizado": mascara_s,  # puede ser None
        "acentuacion": acentuacion,
        "espectro_acentuacion": espectro_a,
        "mascara_acentuacion": mascara_a,
        "gradiente_0": grad_0,
        "gradiente_45": grad_45,
        "gradiente_90": grad_90,
        "gradiente_135": grad_135,
        "gradiente_magnitud": grad_mag,
        "final_binaria": final_binaria,
        "regiones": regiones,
        "bounding_box": bounding_box,
        "imagen_bounding_box": img_bbox,
        "regiones_numeradas": regiones_numeradas,
        "recortes_regiones": recortes,
    }


def guardar_resultado(imagen, ruta):
    Image.fromarray(imagen.astype(np.uint8)).save(ruta)
