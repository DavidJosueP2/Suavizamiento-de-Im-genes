# Image Softening

Aplicacion de escritorio en Python para procesamiento digital de imagenes en escala de grises.

## Funciones

- Carga una imagen desde el computador.
- Convierte la imagen a escala de grises.
- Agrega ruido configurable a la imagen en escala de grises.
- Aplica filtros espaciales manuales 3x3: media, mediana y moda.
- Calcula la FFT 2D centrada de la imagen con ruido.
- Aplica una mascara circular pasa bajas mediante un radio configurable.
- Reconstruye la imagen filtrada con `ifftshift` e `ifft2`.
- Muestra la imagen original en grises, la imagen con ruido, el resultado espacial y el resultado en campo de frecuencia.

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
- `image_processing.py`: carga, escala de grises, ruido, filtros 3x3 y filtro pasa bajas en Fourier.

## Restricciones implementadas

- Los filtros 3x3 estan programados manualmente.
- No se usan funciones listas como `cv2.blur`, `cv2.medianBlur`, `scipy.ndimage`, `skimage.filters` o `PIL.ImageFilter`.
- El padding se hace manualmente por replicacion del borde.
- La Transformada de Fourier se calcula con `np.fft.fft2` y `np.fft.fftshift`.
- La reconstruccion en frecuencia usa `np.fft.ifftshift` y `np.fft.ifft2`.
- No se usan funciones predefinidas de filtrado en frecuencia; la mascara circular se implementa manualmente.
