# Image Softening

Aplicacion de escritorio en Python para procesamiento digital de imagenes a color.

## Funciones

- Carga una imagen desde el computador.
- Trabaja con la imagen RGB a color.
- Permite activar preprocesamiento: escala de grises como entrada del flujo.
- El flujo principal ahora sigue: escala de grises, ruido y normalizacion.
- Agrega ruido sal y pimienta configurable sobre la imagen en escala de grises.
- Aplica suavizado en dominio espacial o de frecuencia.
- En dominio espacial permite media, mediana y moda.
- Permite seleccionar mascaras impares: 3x3, 5x5, 7x7, 9x9 y 11x11.
- Aplica acentuado en dominio espacial o de frecuencia.
- En acentuado espacial usa Laplaciano.
- En frecuencia muestra el espectro FFT y la mascara aplicada.
- Reconoce bordes por gradientes con Prewitt o Sobel.
- Calcula la FFT 2D centrada para filtros pasa bajas y pasa altas.
- Reconstruye la imagen filtrada con `ifftshift` e `ifft2`.
- Aplica acentuacion de bordes, binariza la imagen final del flujo y muestra regiones numeradas con recortes individuales.

## Instalacion

```bash
python -m pip install -r requirements.txt
```

## Ejecucion

```bash
python app.py
```

## Archivos

- `app.py`: interfaz grafica con opciones por seccion y barra de progreso.
- `filtros.py`: carga, preprocesamiento, ruido, suavizado, acentuado, Fourier, gradientes, binarizacion y regiones.

## Restricciones implementadas

- Los filtros espaciales estan programados manualmente.
- Laplaciano, Prewitt, Sobel y high-pass estan implementados en el proyecto.
- No se usan funciones listas como `cv2.blur`, `cv2.medianBlur`, `scipy.ndimage`, `skimage.filters` o `PIL.ImageFilter`.
- El padding se hace manualmente por replicacion del borde.
- La normalizacion min-max y la binarizacion se implementan manualmente.
- La Transformada de Fourier se calcula con `np.fft.fft2` y `np.fft.fftshift`.
- La reconstruccion en frecuencia usa `np.fft.ifftshift` y `np.fft.ifft2`.
- No se usan funciones predefinidas de filtrado en frecuencia.
