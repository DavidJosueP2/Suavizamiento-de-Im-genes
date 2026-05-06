import os
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw

from filtros import cargar_imagen, procesar_imagen, actualizar_preprocesamiento


APP_TITLE = "Procesamiento Digital de Imagenes"
PANEL_IMAGE_SIZE = (290, 210)


def matriz_a_imagen_pil(imagen_matriz):
    matriz = imagen_matriz.astype(np.uint8)
    if matriz.ndim == 2:
        return Image.fromarray(matriz, mode="L").convert("RGB")
    return Image.fromarray(matriz, mode="RGB")


def histograma_a_imagen(histograma, size=PANEL_IMAGE_SIZE):
    ancho, alto = size
    margen = 12
    imagen = Image.new("RGB", size, "#ffffff")
    dibujo = ImageDraw.Draw(imagen)
    maximo = int(histograma.max()) if histograma.size else 0

    dibujo.line((margen, alto - margen, ancho - margen, alto - margen), fill="#111827")
    dibujo.line((margen, margen, margen, alto - margen), fill="#111827")

    if maximo == 0:
        return imagen

    ancho_util = ancho - 2 * margen
    alto_util = alto - 2 * margen
    for intensidad in range(256):
        x = margen + int((intensidad / 255) * (ancho_util - 1))
        altura = int((int(histograma[intensidad]) / maximo) * alto_util)
        dibujo.line((x, alto - margen, x, alto - margen - altura), fill="#2563eb")

    return imagen


def preparar_imagen_panel(imagen_pil, max_size=PANEL_IMAGE_SIZE):
    copia = imagen_pil.copy()
    copia.thumbnail(max_size, Image.Resampling.LANCZOS)
    fondo = Image.new("RGB", max_size, "#ffffff")
    x = (max_size[0] - copia.width) // 2
    y = (max_size[1] - copia.height) // 2
    fondo.paste(copia.convert("RGB"), (x, y))
    return fondo


class ImageSofteningApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x780")
        self.minsize(980, 680)
        self._maximizar_ventana()

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.ruta_imagen = None
        self.imagen_color = None
        self.resultados = {}
        self.paneles = {}
        self.ctk_images = {}
        self._imagenes_recientes = []
        self.preprocesamiento_activo = ctk.BooleanVar(value=False)

        self._crear_interfaz()

    def _maximizar_ventana(self):
        try:
            self.state("zoomed")
        except Exception:
            pass

    def _crear_interfaz(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        encabezado = ctk.CTkFrame(self, fg_color="#e5e7eb", corner_radius=0)
        encabezado.grid(row=0, column=0, sticky="ew")
        encabezado.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            encabezado,
            text="Cargar imagen",
            command=self.cargar_imagen,
            width=150,
            height=34,
            fg_color="#d1d5db",
            hover_color="#c4c9d2",
            text_color="#111827",
        ).grid(row=0, column=0, padx=18, pady=12)

        self.lbl_archivo = ctk.CTkLabel(
            encabezado, text="Sin imagen cargada", text_color="#4b5563", anchor="w"
        )
        self.lbl_archivo.grid(row=0, column=1, sticky="ew", padx=8)

        ctk.CTkButton(
            encabezado,
            text="Procesar flujo completo",
            command=self.procesar_imagen,
            width=180,
            height=34,
            fg_color="#16a34a",
            hover_color="#15803d",
            text_color="#ffffff",
        ).grid(row=0, column=2, padx=8, pady=12)

        self.barra_progreso = ctk.CTkProgressBar(encabezado, width=160, height=10)
        self.barra_progreso.set(0)
        self.barra_progreso.grid(row=0, column=3, padx=(8, 4), pady=12)

        self.lbl_progreso = ctk.CTkLabel(encabezado, text="0 %", text_color="#4b5563", width=44)
        self.lbl_progreso.grid(row=0, column=4, padx=(0, 18), pady=12)

        self.contenedor = ctk.CTkScrollableFrame(self, fg_color="#ffffff", corner_radius=0)
        self.contenedor.grid(row=1, column=0, sticky="nsew")
        self.contenedor.grid_columnconfigure((0, 1, 2, 3), weight=1)

        fila = self._crear_seccion_preprocesamiento(0)
        fila = self._crear_seccion_ruido(fila)
        fila = self._crear_seccion_espacial(fila)
        fila = self._crear_seccion_frecuencia(fila)
        self._crear_seccion_canny(fila)

    def _crear_titulo_seccion(self, titulo, fila):
        ctk.CTkLabel(
            self.contenedor,
            text=titulo,
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="#111827",
            anchor="w",
        ).grid(row=fila, column=0, columnspan=4, sticky="ew", padx=22, pady=(24, 6))
        return fila + 1

    def _crear_panel(self, clave, titulo, fila, columna):
        panel = ctk.CTkFrame(self.contenedor, fg_color="#ffffff", corner_radius=0)
        panel.grid(row=fila, column=columna, sticky="nsew", padx=0, pady=0)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel,
            text=titulo,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#111827",
            anchor="center",
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 6))

        etiqueta = ctk.CTkLabel(panel, text="Sin imagen", text_color="#9ca3af", fg_color="#ffffff")
        etiqueta.grid(row=1, column=0, padx=10, pady=(0, 16), sticky="ew")
        self.paneles[clave] = etiqueta

    def _crear_slider(self, padre, texto, desde, hasta, valor, comando):
        etiqueta = ctk.CTkLabel(padre, text=f"{texto}: {int(valor)}", text_color="#111827", anchor="w")
        etiqueta.pack(fill="x", padx=8)
        slider = ctk.CTkSlider(
            padre,
            from_=desde,
            to=hasta,
            number_of_steps=max(1, int(hasta - desde)),
            command=comando,
        )
        slider.set(valor)
        slider.pack(fill="x", padx=8, pady=(2, 8))
        return slider, etiqueta

    def _crear_controles(self, fila, columna, titulo):
        frame = ctk.CTkFrame(self.contenedor, fg_color="#f3f4f6", corner_radius=8)
        frame.grid(row=fila, column=columna, sticky="nsew", padx=12, pady=8)
        ctk.CTkLabel(
            frame,
            text=titulo,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#111827",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(10, 8))
        return frame

    def _crear_seccion_preprocesamiento(self, fila):
        fila = self._crear_titulo_seccion("Seccion 1: Preprocesamiento", fila)
        controles = self._crear_controles(fila, 0, "Opciones")

        ctk.CTkCheckBox(
            controles,
            text="Aplicar preprocesamiento al flujo",
            variable=self.preprocesamiento_activo,
            command=self._al_cambiar_preprocesamiento,
            text_color="#111827",
        ).pack(fill="x", padx=8, pady=(0, 8))

        self.slider_min, self.lbl_min = self._crear_slider(
            controles, "Minimo", 0, 255, 0, self._al_cambiar_preprocesamiento_slider
        )
        self.slider_max, self.lbl_max = self._crear_slider(
            controles, "Maximo", 0, 255, 255, self._al_cambiar_preprocesamiento_slider
        )
        self.slider_umbral_binario, self.lbl_umbral_binario = self._crear_slider(
            controles, "Umbral binario final", 0, 255, 128, self._actualizar_lbl_umbral_binario
        )

        self._crear_panel("original", "Imagen original RGB", fila, 1)
        self._crear_panel("gris", "Escala de grises", fila, 2)
        self._crear_panel("normalizada", "Imagen normalizada", fila, 3)
        self._crear_panel("hist_original", "Histograma original", fila + 1, 1)
        self._crear_panel("hist_normalizado", "Histograma normalizado", fila + 1, 2)
        self._crear_panel("pre_binaria", "Binaria por umbral", fila + 1, 3)
        return fila + 2

    def _crear_seccion_ruido(self, fila):
        fila = self._crear_titulo_seccion("Seccion 2: Ruido", fila)
        controles = self._crear_controles(fila, 0, "Opciones de ruido")
        self.slider_ruido, self.lbl_ruido = self._crear_slider(
            controles, "Porcentaje de ruido", 0, 50, 30, self._actualizar_lbl_ruido
        )
        self._crear_panel("base", "Imagen de entrada al proceso", fila, 1)
        self._crear_panel("ruido", "Imagen con ruido sal y pimienta", fila, 2)
        return fila + 1

    def _crear_seccion_espacial(self, fila):
        fila = self._crear_titulo_seccion("Seccion 3: Filtros espaciales", fila)
        controles = self._crear_controles(fila, 0, "Opciones espaciales")

        ctk.CTkLabel(controles, text="Filtro", text_color="#111827", anchor="w").pack(fill="x", padx=8)
        self.selector_filtro = ctk.CTkOptionMenu(
            controles,
            values=[
                "Media",
                "Mediana",
                "Moda",
                "High-pass",
                "Sobel horizontal",
                "Sobel vertical",
                "Sobel combinado",
            ],
        )
        self.selector_filtro.set("Mediana")
        self.selector_filtro.pack(fill="x", padx=8, pady=(2, 8))

        ctk.CTkLabel(controles, text="Tamano de mascara", text_color="#111827", anchor="w").pack(fill="x", padx=8)
        self.selector_mascara = ctk.CTkOptionMenu(
            controles, values=["3x3", "5x5", "7x7", "9x9", "11x11"]
        )
        self.selector_mascara.set("3x3")
        self.selector_mascara.pack(fill="x", padx=8, pady=(2, 8))

        self._crear_panel("espacial_entrada", "Entrada espacial", fila, 1)
        self._crear_panel("espacial", "Resultado espacial", fila, 2)
        return fila + 1

    def _crear_seccion_frecuencia(self, fila):
        fila = self._crear_titulo_seccion("Seccion 4: Filtro en frecuencia", fila)
        controles = self._crear_controles(fila, 0, "Opciones de frecuencia")

        ctk.CTkLabel(controles, text="Tipo de filtro", text_color="#111827", anchor="w").pack(fill="x", padx=8)
        self.selector_frecuencia = ctk.CTkOptionMenu(controles, values=["Pasa bajas", "Pasa altas"])
        self.selector_frecuencia.set("Pasa bajas")
        self.selector_frecuencia.pack(fill="x", padx=8, pady=(2, 8))

        self.slider_radio, self.lbl_radio = self._crear_slider(
            controles, "Radio Fourier", 1, 300, 60, self._actualizar_lbl_radio
        )

        self._crear_panel("espectro_original", "Espectro original", fila, 1)
        self._crear_panel("espectro_filtrado", "Espectro filtrado", fila, 2)
        self._crear_panel("frecuencia", "Imagen reconstruida", fila, 3)
        return fila + 1

    def _crear_seccion_canny(self, fila):
        fila = self._crear_titulo_seccion("Seccion 5: Canny + histeresis", fila)
        controles = self._crear_controles(fila, 0, "Opciones Canny")
        self.slider_canny_bajo, self.lbl_canny_bajo = self._crear_slider(
            controles, "Umbral bajo", 0, 255, 50, self._actualizar_lbl_canny_bajo
        )
        self.slider_canny_alto, self.lbl_canny_alto = self._crear_slider(
            controles, "Umbral alto", 0, 255, 120, self._actualizar_lbl_canny_alto
        )

        self._crear_panel("canny_magnitud", "Magnitud de gradiente", fila, 1)
        self._crear_panel("bordes", "Bordes Canny", fila, 2)
        self._crear_panel("final_binaria", "Imagen final binaria", fila, 3)

    def _actualizar_lbl_ruido(self, valor):
        self.lbl_ruido.configure(text=f"Porcentaje de ruido: {int(float(valor))} %")

    def _actualizar_lbl_radio(self, valor):
        self.lbl_radio.configure(text=f"Radio Fourier: {int(float(valor))} px")

    def _actualizar_lbl_umbral_binario(self, valor):
        self.lbl_umbral_binario.configure(text=f"Umbral binario final: {int(float(valor))}")
        if self.imagen_color is not None:
            self.actualizar_preprocesamiento()

    def _actualizar_lbl_canny_bajo(self, valor):
        self.lbl_canny_bajo.configure(text=f"Umbral bajo: {int(float(valor))}")

    def _actualizar_lbl_canny_alto(self, valor):
        self.lbl_canny_alto.configure(text=f"Umbral alto: {int(float(valor))}")

    def _al_cambiar_preprocesamiento_slider(self, valor):
        self.lbl_min.configure(text=f"Minimo: {int(self.slider_min.get())}")
        self.lbl_max.configure(text=f"Maximo: {int(self.slider_max.get())}")
        if self.imagen_color is not None:
            self.actualizar_preprocesamiento()

    def _al_cambiar_preprocesamiento(self):
        if self.imagen_color is not None:
            self.actualizar_preprocesamiento()
            self.procesar_imagen()

    def _actualizar_progreso(self, valor):
        self.barra_progreso.set(valor)
        self.lbl_progreso.configure(text=f"{int(valor * 100)} %")
        self.update_idletasks()

    def _mostrar_en_panel(self, clave, imagen_pil):
        imagen_panel = preparar_imagen_panel(imagen_pil)
        imagen_ctk = ctk.CTkImage(light_image=imagen_panel, dark_image=imagen_panel, size=PANEL_IMAGE_SIZE)
        anterior = self.ctk_images.get(clave)
        if anterior is not None:
            self._imagenes_recientes.append(anterior)
            self._imagenes_recientes = self._imagenes_recientes[-30:]

        self.ctk_images[clave] = imagen_ctk
        self.paneles[clave].configure(image=imagen_ctk, text="")

    def _mostrar_matriz(self, clave, matriz):
        self._mostrar_en_panel(clave, matriz_a_imagen_pil(matriz))

    def _mostrar_histograma(self, clave, histograma):
        self._mostrar_en_panel(clave, histograma_a_imagen(histograma))

    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imagenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not ruta:
            return

        try:
            self.ruta_imagen = ruta
            self.imagen_color = cargar_imagen(ruta)
            self.resultados = {}

            nombre = os.path.basename(ruta)
            alto, ancho = self.imagen_color.shape[:2]
            self.lbl_archivo.configure(text=f"{nombre}    {ancho} x {alto} px")

            radio_maximo = max(1, min(alto, ancho) // 2)
            self.slider_radio.configure(to=radio_maximo, number_of_steps=radio_maximo)
            self.slider_radio.set(min(60, radio_maximo))
            self._actualizar_lbl_radio(self.slider_radio.get())

            self.actualizar_preprocesamiento()
            self.procesar_imagen()
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{exc}")

    def _valores_preprocesamiento(self):
        minimo = int(self.slider_min.get())
        maximo = int(self.slider_max.get())
        umbral = int(self.slider_umbral_binario.get())
        if maximo <= minimo:
            maximo = min(255, minimo + 1)
            self.slider_max.set(maximo)
            self.lbl_max.configure(text=f"Maximo: {maximo}")
        return minimo, maximo, umbral

    def actualizar_preprocesamiento(self):
        if self.imagen_color is None:
            return

        minimo, maximo, umbral = self._valores_preprocesamiento()
        datos = actualizar_preprocesamiento(self.imagen_color, minimo, maximo, umbral)
        self.resultados["preprocesamiento"] = datos

        self._mostrar_matriz("original", datos["original"])
        self._mostrar_matriz("gris", datos["gris"])
        self._mostrar_matriz("normalizada", datos["normalizada"])
        self._mostrar_matriz("pre_binaria", datos["binaria"])
        self._mostrar_histograma("hist_original", datos["histograma_original"])
        self._mostrar_histograma("hist_normalizado", datos["histograma_normalizado"])

    def procesar_imagen(self):
        if self.imagen_color is None:
            messagebox.showwarning("Imagen requerida", "Primero carga una imagen.")
            return

        try:
            self.configure(cursor="watch")
            self._actualizar_progreso(0)

            minimo, maximo, umbral = self._valores_preprocesamiento()
            tamano_mascara = int(self.selector_mascara.get().split("x")[0])
            canny_bajo = int(self.slider_canny_bajo.get())
            canny_alto = int(self.slider_canny_alto.get())
            if canny_alto <= canny_bajo:
                canny_alto = min(255, canny_bajo + 1)
                self.slider_canny_alto.set(canny_alto)
                self.lbl_canny_alto.configure(text=f"Umbral alto: {canny_alto}")

            resultados = procesar_imagen(
                self.imagen_color,
                int(self.slider_ruido.get()),
                self.selector_filtro.get(),
                tamano_mascara,
                int(self.slider_radio.get()),
                tipo_frecuencia=self.selector_frecuencia.get(),
                aplicar_preprocesamiento=self.preprocesamiento_activo.get(),
                minimo_norm=minimo,
                maximo_norm=maximo,
                umbral_binario=umbral,
                canny_bajo=canny_bajo,
                canny_alto=canny_alto,
                progreso=self._actualizar_progreso,
            )
            self.resultados = resultados

            self._mostrar_matriz("base", resultados["base_proceso"])
            self._mostrar_matriz("ruido", resultados["ruido"])
            self._mostrar_matriz("espacial_entrada", resultados["ruido"])
            self._mostrar_matriz("espacial", resultados["espacial"])
            self._mostrar_matriz("espectro_original", resultados["espectro_original"])
            self._mostrar_matriz("espectro_filtrado", resultados["espectro_filtrado"])
            self._mostrar_matriz("frecuencia", resultados["frecuencia"])
            self._mostrar_matriz("canny_magnitud", resultados["canny_magnitud"])
            self._mostrar_matriz("bordes", resultados["bordes"])
            self._mostrar_matriz("final_binaria", resultados["final_binaria"])
            self._actualizar_progreso(1)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo procesar la imagen:\n{exc}")
        finally:
            self.configure(cursor="")


if __name__ == "__main__":
    app = ImageSofteningApp()
    app.mainloop()
