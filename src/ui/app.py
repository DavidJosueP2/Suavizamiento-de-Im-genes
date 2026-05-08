import os
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image

from src.app.filtros import cargar_imagen, procesar_imagen, actualizar_preprocesamiento


APP_TITLE = "Procesamiento Digital de Imagenes"
PANEL_IMAGE_SIZE = (290, 210)
REGION_MAP_SIZE = (680, 430)
REGION_CROP_SIZE = (150, 110)


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
        self.panel_frames = {}
        self.panel_sizes = {}
        self.ctk_images = {}
        self._imagenes_recientes = []
        self.preprocesamiento_activo = ctk.BooleanVar(value=True)

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
        fila = self._crear_seccion_suavizado(fila)
        fila = self._crear_seccion_acentuacion(fila)
        fila = self._crear_seccion_gradientes(fila)
        self._crear_seccion_regiones(fila)
        self._actualizar_paneles_por_dominio()

    def _crear_titulo_seccion(self, titulo, fila):
        ctk.CTkLabel(
            self.contenedor,
            text=titulo,
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="#111827",
            anchor="w",
        ).grid(row=fila, column=0, columnspan=4, sticky="ew", padx=22, pady=(24, 6))
        return fila + 1

    def _crear_panel(self, clave, titulo, fila, columna, columnspan=1, image_size=PANEL_IMAGE_SIZE):
        panel = ctk.CTkFrame(self.contenedor, fg_color="#ffffff", corner_radius=0)
        panel.grid(row=fila, column=columna, columnspan=columnspan, sticky="nsew", padx=0, pady=0)
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
        self.panel_frames[clave] = panel
        self.panel_sizes[clave] = image_size

    def _crear_slider(self, padre, texto, desde, hasta, valor, comando):
        grupo = ctk.CTkFrame(padre, fg_color="transparent")
        grupo.pack(side="left", padx=8, pady=6)
        etiqueta = ctk.CTkLabel(grupo, text=f"{texto}: {int(valor)}", text_color="#111827", anchor="w")
        etiqueta.pack(fill="x")
        slider = ctk.CTkSlider(
            grupo,
            from_=desde,
            to=hasta,
            number_of_steps=max(1, int(hasta - desde)),
            command=comando,
            width=220,
        )
        slider.set(valor)
        slider.pack(fill="x", pady=(2, 0))
        return slider, etiqueta, grupo

    def _crear_selector(self, padre, texto, valores, valor):
        grupo = ctk.CTkFrame(padre, fg_color="transparent")
        grupo.pack(side="left", padx=8, pady=6)
        ctk.CTkLabel(grupo, text=texto, text_color="#111827", anchor="w").pack(fill="x")
        selector = ctk.CTkOptionMenu(grupo, values=valores, width=170)
        selector.set(valor)
        selector.pack(fill="x", pady=(2, 0))
        return selector, grupo

    def _crear_controles(self, fila):
        frame = ctk.CTkFrame(self.contenedor, fg_color="#f3f4f6", corner_radius=6)
        frame.grid(row=fila, column=0, columnspan=4, sticky="w", padx=28, pady=(4, 8))
        return frame

    def _crear_seccion_preprocesamiento(self, fila):
        fila = self._crear_titulo_seccion("Seccion 1: Preprocesamiento", fila)
        controles = self._crear_controles(fila)

        ctk.CTkCheckBox(
            controles,
            text="Aplicar preprocesamiento al flujo",
            variable=self.preprocesamiento_activo,
            command=self._al_cambiar_preprocesamiento,
            text_color="#111827",
        ).pack(side="left", padx=10, pady=8)

        self.slider_min, self.lbl_min, self.ctrl_min = self._crear_slider(
            controles, "Minimo", 0, 255, 0, self._al_cambiar_preprocesamiento_slider
        )
        self.slider_max, self.lbl_max, self.ctrl_max = self._crear_slider(
            controles, "Maximo", 0, 255, 255, self._al_cambiar_preprocesamiento_slider
        )
        self._crear_panel("original", "Imagen original RGB", fila + 1, 0)
        self._crear_panel("gris", "Escala de grises", fila + 1, 1)
        return fila + 2

    def _crear_seccion_ruido(self, fila):
        fila = self._crear_titulo_seccion("Seccion 2: Ruido", fila)
        controles = self._crear_controles(fila)
        self.slider_ruido, self.lbl_ruido, self.ctrl_ruido = self._crear_slider(
            controles, "Porcentaje de ruido", 0, 50, 30, self._actualizar_lbl_ruido
        )
        self._crear_panel("base", "Entrada en grises", fila + 1, 0)
        self._crear_panel("ruido", "Imagen con ruido", fila + 1, 1)
        self._crear_panel("normalizada_flujo", "Normalizacion despues del ruido", fila + 1, 2)
        return fila + 2

    def _crear_seccion_suavizado(self, fila):
        fila = self._crear_titulo_seccion("Seccion 3: Suavizado", fila)
        controles = self._crear_controles(fila)
        self.selector_dominio_suavizado, self.ctrl_dominio_suavizado = self._crear_selector(
            controles, "Dominio", ["Espacial", "Frecuencia"], "Espacial"
        )
        self.selector_dominio_suavizado.configure(command=self._al_cambiar_dominio_suavizado)
        self.selector_suavizado, self.ctrl_suavizado = self._crear_selector(
            controles, "Filtro seleccionable", ["Media", "Mediana", "Moda"], "Mediana"
        )
        self.selector_mascara, self.ctrl_mascara = self._crear_selector(
            controles, "Mascara", ["3x3", "5x5", "7x7", "9x9", "11x11"], "3x3"
        )
        self.slider_radio_suavizado, self.lbl_radio_suavizado, self.ctrl_radio_suavizado = self._crear_slider(
            controles, "Radio pasa bajas", 1, 300, 60, self._actualizar_lbl_radio_suavizado
        )

        self._crear_panel("suavizado_entrada", "Entrada a suavizado", fila + 1, 0)
        self._crear_panel("suavizado", "Resultado suavizado", fila + 1, 1)
        self._crear_panel("espectro_suavizado", "Espectro FFT", fila + 1, 2)
        self._crear_panel("mascara_suavizado", "Mascara pasa bajas", fila + 1, 3)
        return fila + 2

    def _crear_seccion_acentuacion(self, fila):
        fila = self._crear_titulo_seccion("Seccion 4: Acentuado", fila)
        controles = self._crear_controles(fila)
        self.selector_dominio_acentuacion, self.ctrl_dominio_acentuacion = self._crear_selector(
            controles, "Dominio", ["Espacial", "Frecuencia"], "Espacial"
        )
        self.selector_dominio_acentuacion.configure(command=self._al_cambiar_dominio_acentuacion)
        self.selector_acentuacion, self.ctrl_acentuacion = self._crear_selector(
            controles, "Filtro seleccionable", ["Laplaciano"], "Laplaciano"
        )

        self.slider_radio_acentuacion, self.lbl_radio_acentuacion, self.ctrl_radio_acentuacion = self._crear_slider(
            controles, "Radio pasa altas", 1, 300, 60, self._actualizar_lbl_radio_acentuacion
        )

        self._crear_panel("acentuacion_entrada", "Entrada a acentuacion", fila + 1, 0)
        self._crear_panel("acentuacion", "Resultado acentuado", fila + 1, 1)
        self._crear_panel("espectro_acentuacion", "Espectro FFT", fila + 1, 2)
        self._crear_panel("mascara_acentuacion", "Mascara pasa altas", fila + 1, 3)
        return fila + 2

    def _crear_seccion_gradientes(self, fila):
        fila = self._crear_titulo_seccion("Seccion 5: Reconocimiento de bordes por gradientes", fila)
        controles = self._crear_controles(fila)
        self.selector_gradiente, self.ctrl_gradiente = self._crear_selector(
            controles, "Metodo", ["Prewitt", "Sobel"], "Prewitt"
        )

        self._crear_panel("gradiente_x", "Gradiente X", fila + 1, 0)
        self._crear_panel("gradiente_y", "Gradiente Y", fila + 1, 1)
        self._crear_panel("gradiente_magnitud", "Magnitud del gradiente", fila + 1, 2)
        return fila + 2

    def _crear_seccion_regiones(self, fila):
        fila = self._crear_titulo_seccion("Seccion 6: Binarizacion y deteccion de regiones", fila)
        controles = self._crear_controles(fila)
        self.slider_umbral_binario, self.lbl_umbral_binario, self.ctrl_umbral_binario = self._crear_slider(
            controles, "Umbral binario", 0, 255, 60, self._actualizar_lbl_umbral_binario
        )
        self.slider_area_minima, self.lbl_area_minima, self.ctrl_area_minima = self._crear_slider(
            controles, "Area minima", 1, 2000, 10, self._actualizar_lbl_area_minima
        )

        self._crear_panel("final_binaria", "Imagen final binaria", fila + 1, 0)
        self._crear_panel(
            "regiones_numeradas",
            "Imagen final con zonas numeradas",
            fila + 1,
            1,
            columnspan=3,
            image_size=REGION_MAP_SIZE,
        )
        fila += 2

        panel = ctk.CTkFrame(self.contenedor, fg_color="#ffffff", corner_radius=0)
        panel.grid(row=fila, column=0, columnspan=4, sticky="nsew", padx=0, pady=0)
        panel.grid_columnconfigure((0, 1, 2, 3), weight=1)
        ctk.CTkLabel(
            panel,
            text="Zonas individuales",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#111827",
            anchor="w",
        ).grid(row=0, column=0, columnspan=4, sticky="ew", padx=28, pady=(10, 4))

        self.frame_recortes = ctk.CTkScrollableFrame(
            panel, fg_color="#ffffff", corner_radius=0, height=330
        )
        self.frame_recortes.grid(row=1, column=0, columnspan=4, sticky="ew", padx=24, pady=(0, 16))
        self.frame_recortes.grid_columnconfigure((0, 1, 2, 3), weight=1)
        ctk.CTkLabel(self.frame_recortes, text="Sin regiones", text_color="#9ca3af").grid(
            row=0, column=0, columnspan=4, pady=20
        )
        return fila + 1

    def _actualizar_lbl_ruido(self, valor):
        self.lbl_ruido.configure(text=f"Porcentaje de ruido: {int(float(valor))} %")

    def _actualizar_lbl_radio_suavizado(self, valor):
        self.lbl_radio_suavizado.configure(text=f"Radio pasa bajas: {int(float(valor))} px")

    def _actualizar_lbl_radio_acentuacion(self, valor):
        self.lbl_radio_acentuacion.configure(text=f"Radio pasa altas: {int(float(valor))} px")

    def _actualizar_lbl_umbral_binario(self, valor):
        self.lbl_umbral_binario.configure(text=f"Umbral binario: {int(float(valor))}")

    def _actualizar_lbl_area_minima(self, valor):
        self.lbl_area_minima.configure(text=f"Area minima: {int(float(valor))} px")

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
        size = self.panel_sizes.get(clave, PANEL_IMAGE_SIZE)
        imagen_panel = preparar_imagen_panel(imagen_pil, size)
        imagen_ctk = ctk.CTkImage(light_image=imagen_panel, dark_image=imagen_panel, size=size)
        anterior = self.ctk_images.get(clave)
        if anterior is not None:
            self._imagenes_recientes.append(anterior)
            self._imagenes_recientes = self._imagenes_recientes[-30:]

        self.ctk_images[clave] = imagen_ctk
        self.paneles[clave].configure(image=imagen_ctk, text="")

    def _mostrar_matriz(self, clave, matriz):
        self._mostrar_en_panel(clave, matriz_a_imagen_pil(matriz))

    def _mostrar_panel(self, clave, visible):
        frame = self.panel_frames.get(clave)
        if frame is None:
            return
        if visible:
            frame.grid()
        else:
            frame.grid_remove()

    def _mostrar_control(self, control, visible):
        if visible:
            control.pack(side="left", padx=8, pady=6)
        else:
            control.pack_forget()

    def _actualizar_paneles_por_dominio(self):
        suavizado_frecuencia = self.selector_dominio_suavizado.get() == "Frecuencia"
        acentuacion_frecuencia = self.selector_dominio_acentuacion.get() == "Frecuencia"

        self._mostrar_panel("espectro_suavizado", suavizado_frecuencia)
        self._mostrar_panel("mascara_suavizado", suavizado_frecuencia)
        self._mostrar_panel("espectro_acentuacion", acentuacion_frecuencia)
        self._mostrar_panel("mascara_acentuacion", acentuacion_frecuencia)
        self._mostrar_control(self.ctrl_suavizado, True)
        self._mostrar_control(self.ctrl_mascara, not suavizado_frecuencia)
        self._mostrar_control(self.ctrl_radio_suavizado, suavizado_frecuencia)
        self._mostrar_control(self.ctrl_acentuacion, not acentuacion_frecuencia)
        self._mostrar_control(self.ctrl_radio_acentuacion, acentuacion_frecuencia)

    def _al_cambiar_dominio_suavizado(self, valor):
        if valor == "Frecuencia":
            self.selector_suavizado.configure(values=["Pasa bajas", "Pasa bajas gausiano"])
            self.selector_suavizado.set("Pasa bajas")
        else:
            self.selector_suavizado.configure(values=["Media", "Mediana", "Moda"])
            self.selector_suavizado.set("Mediana")
        self._actualizar_paneles_por_dominio()

    def _al_cambiar_dominio_acentuacion(self, valor):
        if valor == "Frecuencia":
            self.selector_acentuacion.configure(values=["Pasa altas"])
            self.selector_acentuacion.set("Pasa altas")
        else:
            self.selector_acentuacion.configure(values=["Laplaciano"])
            self.selector_acentuacion.set("Laplaciano")
        self._actualizar_paneles_por_dominio()

    def _mostrar_recortes_regiones(self, recortes):
        for widget in self.frame_recortes.winfo_children():
            widget.destroy()

        if not recortes:
            ctk.CTkLabel(self.frame_recortes, text="Sin regiones", text_color="#9ca3af").grid(
                row=0, column=0, pady=20
            )
            return

        for indice, item in enumerate(recortes):
            region = item["region"]
            imagen = matriz_a_imagen_pil(item["imagen"])
            imagen_panel = preparar_imagen_panel(imagen, REGION_CROP_SIZE)
            imagen_ctk = ctk.CTkImage(
                light_image=imagen_panel,
                dark_image=imagen_panel,
                size=REGION_CROP_SIZE,
            )
            self.ctk_images[f"region_{region['label']}"] = imagen_ctk

            texto = (
                f"Zona {region['label']}  "
                f"{region['width']}x{region['height']} px  "
                f"Area {region['area']}"
            )
            fila = indice // 4
            columna = indice % 4
            tarjeta = ctk.CTkFrame(self.frame_recortes, fg_color="#ffffff", corner_radius=0)
            tarjeta.grid(row=fila, column=columna, sticky="n", padx=10, pady=(4, 14))
            ctk.CTkLabel(
                tarjeta,
                text=texto,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#111827",
                anchor="center",
            ).grid(row=0, column=0, sticky="ew", padx=4, pady=(2, 4))
            ctk.CTkLabel(tarjeta, image=imagen_ctk, text="", fg_color="#ffffff").grid(
                row=1, column=0, padx=4, pady=(0, 4)
            )

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
            self.slider_radio_suavizado.configure(to=radio_maximo, number_of_steps=radio_maximo)
            self.slider_radio_acentuacion.configure(to=radio_maximo, number_of_steps=radio_maximo)
            self.slider_radio_suavizado.set(min(60, radio_maximo))
            self.slider_radio_acentuacion.set(min(60, radio_maximo))
            self._actualizar_lbl_radio_suavizado(self.slider_radio_suavizado.get())
            self._actualizar_lbl_radio_acentuacion(self.slider_radio_acentuacion.get())

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

    def procesar_imagen(self):
        if self.imagen_color is None:
            messagebox.showwarning("Imagen requerida", "Primero carga una imagen.")
            return

        try:
            self.configure(cursor="watch")
            self._actualizar_progreso(0)

            minimo, maximo, umbral = self._valores_preprocesamiento()
            tamano_mascara = int(self.selector_mascara.get().split("x")[0])
            area_minima = int(self.slider_area_minima.get())

            resultados = procesar_imagen(
                self.imagen_color,
                int(self.slider_ruido.get()),
                self.selector_suavizado.get(),
                tamano_mascara,
                int(self.slider_radio_suavizado.get()),
                dominio_suavizado=self.selector_dominio_suavizado.get(),
                dominio_acentuacion=self.selector_dominio_acentuacion.get(),
                tipo_acentuacion="Laplaciano",
                radio_acentuacion=int(self.slider_radio_acentuacion.get()),
                metodo_gradiente=self.selector_gradiente.get(),
                aplicar_preprocesamiento=self.preprocesamiento_activo.get(),
                minimo_norm=minimo,
                maximo_norm=maximo,
                umbral_binario=umbral,
                min_area_region=area_minima,
                progreso=self._actualizar_progreso,
            )
            self.resultados = resultados
            self._actualizar_paneles_por_dominio()

            self._mostrar_matriz("base", resultados["base_proceso"])
            self._mostrar_matriz("ruido", resultados["ruido"])
            self._mostrar_matriz("normalizada_flujo", resultados["normalizada_flujo"])
            self._mostrar_matriz("suavizado_entrada", resultados["normalizada_flujo"])
            self._mostrar_matriz("suavizado", resultados["suavizado"])
            if resultados["espectro_suavizado"] is not None:
                self._mostrar_matriz("espectro_suavizado", resultados["espectro_suavizado"])
                self._mostrar_matriz("mascara_suavizado", resultados["mascara_suavizado"])
            self._mostrar_matriz("acentuacion_entrada", resultados["suavizado"])
            self._mostrar_matriz("acentuacion", resultados["acentuacion"])
            if resultados["espectro_acentuacion"] is not None:
                self._mostrar_matriz("espectro_acentuacion", resultados["espectro_acentuacion"])
                self._mostrar_matriz("mascara_acentuacion", resultados["mascara_acentuacion"])
            self._mostrar_matriz("gradiente_x", resultados["gradiente_x"])
            self._mostrar_matriz("gradiente_y", resultados["gradiente_y"])
            self._mostrar_matriz("gradiente_magnitud", resultados["gradiente_magnitud"])
            self._mostrar_matriz("final_binaria", resultados["final_binaria"])
            self._mostrar_matriz("regiones_numeradas", resultados["regiones_numeradas"])
            self._mostrar_recortes_regiones(resultados["recortes_regiones"])
            self._actualizar_progreso(1)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo procesar la imagen:\n{exc}")
        finally:
            self.configure(cursor="")


if __name__ == "__main__":
    app = ImageSofteningApp()
    app.mainloop()
