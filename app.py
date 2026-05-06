import os
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image

from filtros import cargar_imagen, procesar_imagen, actualizar_preprocesamiento


APP_TITLE = "Procesamiento Digital de Imagenes"
PANEL_IMAGE_SIZE = (300, 220)


def matriz_a_imagen_pil(imagen_matriz):
    matriz = imagen_matriz.astype(np.uint8)
    if matriz.ndim == 2:
        return Image.fromarray(matriz, mode="L").convert("RGB")
    return Image.fromarray(matriz, mode="RGB")


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
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        barra = ctk.CTkScrollableFrame(
            self, width=270, corner_radius=0, fg_color="#e5e7eb"
        )
        barra.grid(row=0, column=0, sticky="nsew")
        self.grid_columnconfigure(0, minsize=270)

        ctk.CTkButton(
            barra,
            text="Cargar imagen",
            command=self.cargar_imagen,
            height=38,
            fg_color="#d1d5db",
            hover_color="#c4c9d2",
            text_color="#111827",
        ).pack(fill="x", padx=18, pady=(22, 12))

        self.lbl_archivo = ctk.CTkLabel(
            barra, text="Sin imagen cargada", text_color="#4b5563", wraplength=220
        )
        self.lbl_archivo.pack(fill="x", padx=18, pady=(0, 12))

        ctk.CTkLabel(
            barra,
            text="Preprocesamiento de imagen",
            text_color="#111827",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(0, 6))

        self.chk_preprocesamiento = ctk.CTkCheckBox(
            barra,
            text="Aplicar preprocesamiento",
            variable=self.preprocesamiento_activo,
            command=self._al_cambiar_preprocesamiento,
            text_color="#111827",
        )
        self.chk_preprocesamiento.pack(fill="x", padx=18, pady=(0, 10))

        self.slider_min, self.lbl_min = self._crear_slider(
            barra,
            "Minimo de normalizacion",
            0,
            255,
            0,
            self._al_cambiar_preprocesamiento_slider,
        )
        self.slider_max, self.lbl_max = self._crear_slider(
            barra,
            "Maximo de normalizacion",
            0,
            255,
            255,
            self._al_cambiar_preprocesamiento_slider,
        )
        self.slider_umbral, self.lbl_umbral = self._crear_slider(
            barra,
            "Umbral de binarizacion",
            0,
            255,
            128,
            self._al_cambiar_preprocesamiento_slider,
        )

        self.slider_ruido, self.lbl_ruido = self._crear_slider(
            barra, "Porcentaje de ruido", 0, 50, 30, self._actualizar_lbl_ruido
        )

        ctk.CTkLabel(
            barra, text="Filtro espacial", text_color="#111827", anchor="w"
        ).pack(fill="x", padx=18)
        self.selector_filtro = ctk.CTkOptionMenu(
            barra, values=["Media", "Mediana", "Moda"]
        )
        self.selector_filtro.set("Mediana")
        self.selector_filtro.pack(fill="x", padx=18, pady=(4, 10))

        ctk.CTkLabel(
            barra, text="Tamano de mascara", text_color="#111827", anchor="w"
        ).pack(fill="x", padx=18)
        self.selector_mascara = ctk.CTkOptionMenu(
            barra, values=["3x3", "5x5", "7x7", "9x9", "11x11"]
        )
        self.selector_mascara.set("3x3")
        self.selector_mascara.pack(fill="x", padx=18, pady=(4, 10))

        self.slider_radio, self.lbl_radio = self._crear_slider(
            barra, "Radio de Fourier", 1, 300, 60, self._actualizar_lbl_radio
        )

        ctk.CTkButton(
            barra,
            text="Procesar",
            command=self.procesar_imagen,
            height=38,
            fg_color="#16a34a",
            hover_color="#15803d",
            text_color="#ffffff",
        ).pack(fill="x", padx=18, pady=(2, 8))

        self.barra_progreso = ctk.CTkProgressBar(barra, height=10)
        self.barra_progreso.set(0)
        self.barra_progreso.pack(fill="x", padx=18, pady=(0, 4))

        self.lbl_progreso = ctk.CTkLabel(
            barra, text="0 %", text_color="#4b5563", anchor="center"
        )
        self.lbl_progreso.pack(fill="x", padx=18, pady=(0, 12))

        contenedor = ctk.CTkScrollableFrame(self, fg_color="#ffffff", corner_radius=0)
        contenedor.grid(row=0, column=1, sticky="nsew")
        contenedor.grid_columnconfigure((0, 1, 2), weight=1)

        fila = self._crear_seccion(
            contenedor, "Seccion 1: Preprocesamiento opcional", 0
        )
        self._crear_panel(contenedor, "original", "Imagen original RGB", fila, 0)
        self._crear_panel(contenedor, "gris", "Escala de grises", fila, 1)
        self._crear_panel(contenedor, "normalizada", "Imagen normalizada", fila, 2)
        self._crear_panel(contenedor, "binaria", "Imagen binarizada", fila + 1, 0)

        fila = self._crear_seccion(contenedor, "Seccion 2: Ruido", fila + 2)
        self._crear_panel(contenedor, "base", "Imagen base del proceso", fila, 0)
        self._crear_panel(contenedor, "convolucion", "Imagen convolucionada", fila, 1)
        self._crear_panel(contenedor, "ruido", "Imagen con ruido", fila, 2)

        fila = self._crear_seccion(
            contenedor, "Seccion 3: Filtros espacial y frecuencia", fila + 1
        )
        self._crear_panel(contenedor, "espacial", "Filtro espacial", fila, 0)
        self._crear_panel(contenedor, "frecuencia", "Filtro pasa bajo", fila, 1)
        self._crear_panel(contenedor, "espectro", "Espectro FFT", fila, 2)

    def _crear_slider(self, padre, texto, desde, hasta, valor, comando):
        etiqueta = ctk.CTkLabel(
            padre, text=f"{texto}: {int(valor)}", text_color="#111827", anchor="w"
        )
        etiqueta.pack(fill="x", padx=18)
        slider = ctk.CTkSlider(
            padre,
            from_=desde,
            to=hasta,
            number_of_steps=max(1, int(hasta - desde)),
            command=comando,
        )
        slider.set(valor)
        slider.pack(fill="x", padx=18, pady=(2, 10))
        return slider, etiqueta

    def _crear_seccion(self, padre, titulo, fila):
        ctk.CTkLabel(
            padre,
            text=titulo,
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="#111827",
            anchor="w",
        ).grid(row=fila, column=0, columnspan=3, sticky="ew", padx=22, pady=(24, 8))
        return fila + 1

    def _crear_panel(self, padre, clave, titulo, fila, columna):
        panel = ctk.CTkFrame(padre, fg_color="#ffffff", corner_radius=0)
        panel.grid(row=fila, column=columna, padx=0, pady=0, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel,
            text=titulo,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#111827",
            anchor="center",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(8, 6))

        imagen_label = ctk.CTkLabel(
            panel,
            text="Sin imagen",
            text_color="#9ca3af",
            fg_color="#ffffff",
            corner_radius=0,
        )
        imagen_label.grid(row=1, column=0, padx=14, pady=(0, 18), sticky="ew")
        self.paneles[clave] = imagen_label

    def _actualizar_lbl_ruido(self, valor):
        self.lbl_ruido.configure(text=f"Porcentaje de ruido: {int(float(valor))} %")

    def _actualizar_lbl_radio(self, valor):
        self.lbl_radio.configure(text=f"Radio de Fourier: {int(float(valor))} px")

    def _al_cambiar_preprocesamiento_slider(self, valor):
        self.lbl_min.configure(
            text=f"Minimo de normalizacion: {int(self.slider_min.get())}"
        )
        self.lbl_max.configure(
            text=f"Maximo de normalizacion: {int(self.slider_max.get())}"
        )
        self.lbl_umbral.configure(
            text=f"Umbral de binarizacion: {int(self.slider_umbral.get())}"
        )
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
        imagen_ctk = ctk.CTkImage(
            light_image=imagen_panel, dark_image=imagen_panel, size=PANEL_IMAGE_SIZE
        )
        anterior = self.ctk_images.get(clave)
        if anterior is not None:
            self._imagenes_recientes.append(anterior)
            self._imagenes_recientes = self._imagenes_recientes[-20:]

        self.ctk_images[clave] = imagen_ctk
        self.paneles[clave].configure(image=imagen_ctk, text="")

    def _mostrar_matriz(self, clave, matriz):
        self._mostrar_en_panel(clave, matriz_a_imagen_pil(matriz))

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
            self.lbl_archivo.configure(text=f"{nombre}\n{ancho} x {alto} px")

            radio_maximo = max(1, min(alto, ancho) // 2)
            self.slider_radio.configure(to=radio_maximo, number_of_steps=radio_maximo)
            self.slider_radio.set(min(60, radio_maximo))
            self._actualizar_lbl_radio(self.slider_radio.get())

            self._mostrar_matriz("original", self.imagen_color)
            self.actualizar_preprocesamiento()
            self.procesar_imagen()
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{exc}")

    def actualizar_preprocesamiento(self):
        if self.imagen_color is None:
            return

        minimo, maximo, umbral = self._valores_preprocesamiento()
        datos = actualizar_preprocesamiento(self.imagen_color, minimo, maximo, umbral)
        self.resultados["preprocesamiento"] = datos

        self._mostrar_matriz("original", datos["original"])
        self._mostrar_matriz("gris", datos["gris"])
        self._mostrar_matriz("normalizada", datos["normalizada"])
        self._mostrar_matriz("binaria", datos["binaria"])

    def _valores_preprocesamiento(self):
        minimo = int(self.slider_min.get())
        maximo = int(self.slider_max.get())
        umbral = int(self.slider_umbral.get())
        if maximo <= minimo:
            maximo = min(255, minimo + 1)
            self.slider_max.set(maximo)
            self.lbl_max.configure(text=f"Maximo de normalizacion: {maximo}")

        return minimo, maximo, umbral

    def procesar_imagen(self):
        if self.imagen_color is None:
            messagebox.showwarning("Imagen requerida", "Primero carga una imagen.")
            return

        try:
            self.configure(cursor="watch")
            self._actualizar_progreso(0)

            tamano_mascara = int(self.selector_mascara.get().split("x")[0])
            minimo, maximo, umbral = self._valores_preprocesamiento()
            resultados = procesar_imagen(
                self.imagen_color,
                int(self.slider_ruido.get()),
                self.selector_filtro.get(),
                tamano_mascara,
                int(self.slider_radio.get()),
                aplicar_preprocesamiento=self.preprocesamiento_activo.get(),
                minimo_norm=minimo,
                maximo_norm=maximo,
                umbral=umbral,
                progreso=self._actualizar_progreso,
            )
            self.resultados = resultados

            self._mostrar_matriz("base", resultados["base_proceso"])
            self._mostrar_matriz("ruido", resultados["ruido"])
            self._mostrar_matriz("espacial", resultados["espacial"])
            self._mostrar_matriz("frecuencia", resultados["frecuencia"])
            self._mostrar_matriz("espectro", resultados["espectro"])
            self._mostrar_matriz("convolucion", resultados["convolucion"])
            self._actualizar_progreso(1)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo procesar la imagen:\n{exc}")
        finally:
            self.configure(cursor="")


if __name__ == "__main__":
    app = ImageSofteningApp()
    app.mainloop()
