# Image Softening

Aplicacion de escritorio en Python para procesamiento digital de imagenes a color.

## Funciones

- Carga una imagen desde el computador.
- Trabaja con la imagen RGB a color.
- Permite activar preprocesamiento: escala de grises, histograma, normalizacion min-max, histograma normalizado y binarizacion.
- Agrega ruido sal y pimienta configurable sobre la imagen base del proceso.
- Aplica filtros espaciales manuales por canal: media, mediana y moda.
- Permite seleccionar mascaras impares: 3x3, 5x5, 7x7, 9x9 y 11x11.
- Calcula la FFT 2D centrada de la imagen con ruido.
- Muestra el espectro FFT con mascara circular pasa bajo.
- Aplica una mascara circular pasa bajas mediante un radio configurable.
- Reconstruye la imagen filtrada con `ifftshift` e `ifft2`.
- Muestra la imagen original, preprocesamiento, ruido, resultado espacial, espectro y resultado en frecuencia.

## Instalacion

```bash
python -m pip install -r requirements.txt
```

## Ejecucion

```bash
python app.py
```

## Archivos

- `app.py`: interfaz grafica y barra de progreso.
- `filtros.py`: carga, preprocesamiento, ruido, filtros espaciales por canal y filtro pasa bajas en Fourier.

## Restricciones implementadas

- Los filtros espaciales estan programados manualmente.
- No se usan funciones listas como `cv2.blur`, `cv2.medianBlur`, `scipy.ndimage`, `skimage.filters` o `PIL.ImageFilter`.
- El padding se hace manualmente por replicacion del borde.
- La normalizacion min-max y la binarizacion se implementan manualmente.
- La Transformada de Fourier se calcula con `np.fft.fft2` y `np.fft.fftshift`.
- La reconstruccion en frecuencia usa `np.fft.ifftshift` y `np.fft.ifft2`.
- No se usan funciones predefinidas de filtrado en frecuencia; la mascara circular se implementa manualmente.
