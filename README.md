# Image Softening

Aplicacion de escritorio en Python para procesamiento digital de imagenes a color.

## Funciones

- Carga una imagen desde el computador.
- Trabaja con la imagen RGB a color.
- Permite activar preprocesamiento: escala de grises y normalizacion min-max como entrada del flujo.
- Muestra histogramas original y normalizado.
- Agrega ruido sal y pimienta configurable sobre la imagen base del proceso.
- Aplica filtros espaciales por canal: media, mediana, moda, high-pass y Sobel.
- Permite seleccionar mascaras impares: 3x3, 5x5, 7x7, 9x9 y 11x11.
- Calcula la FFT 2D centrada y permite filtro pasa bajas o pasa altas.
- Muestra espectro original, espectro filtrado e imagen reconstruida.
- Reconstruye la imagen filtrada con `ifftshift` e `ifft2`.
- Aplica Canny con histeresis y produce la imagen final binaria.

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
- `filtros.py`: carga, preprocesamiento, ruido, filtros espaciales, Fourier y Canny.

## Restricciones implementadas

- Los filtros espaciales estan programados manualmente.
- Sobel, high-pass, Canny e histeresis estan implementados en el proyecto.
- No se usan funciones listas como `cv2.blur`, `cv2.medianBlur`, `scipy.ndimage`, `skimage.filters` o `PIL.ImageFilter`.
- El padding se hace manualmente por replicacion del borde.
- La normalizacion min-max y la binarizacion se implementan manualmente.
- La Transformada de Fourier se calcula con `np.fft.fft2` y `np.fft.fftshift`.
- La reconstruccion en frecuencia usa `np.fft.ifftshift` y `np.fft.ifft2`.
- No se usan funciones predefinidas de filtrado en frecuencia.
