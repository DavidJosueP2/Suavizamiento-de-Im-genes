from src.app.suavizado_espacial import aplicar_filtro_espacial
from src.app.suavizado_frecuencia import aplicar_filtro_frecuencia_completo


def aplicar_suavizado(imagen, dominio, tipo_suavizado, tamano_mascara, radio):
    if dominio == "Frecuencia":
        if tipo_suavizado == "Pasa bajas gausiano":
            tipo_filtro = "bajas_gaussiano"
        else:
            tipo_filtro = "bajas"
        espectro_orig, espectro_filt, mascara, suavizada = (
            aplicar_filtro_frecuencia_completo(imagen, radio, tipo_filtro)
        )
        return suavizada, espectro_orig, espectro_filt, mascara
    # Dominio espacial
    suavizada = aplicar_filtro_espacial(imagen, tipo_suavizado, tamano_mascara)
    return suavizada, None, None, None
