import os
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from filtros import cargar_imagen, convertir_grises, procesar_imagen


APP_TITLE = "Procesamiento Digital de Imagenes"
PANEL_IMAGE_SIZE = (330, 260)


def matriz_a_imagen_pil(imagen_matriz):
    return Image.fromarray(imagen_matriz.astype("uint8"))


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
        self.state("zoomed")

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.ruta_imagen = None
        self.imagen_original_pil = None
        self.imagen_gris = None
        self.imagen_ruido = None
        self.resultado_espacial = None
        self.resultado_frecuencia = None
        self.paneles = {}
        self.ctk_images = {}

        self._crear_interfaz()

    def _crear_interfaz(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        barra = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color="#e5e7eb")
        barra.grid(row=0, column=0, sticky="nsew")
        barra.grid_propagate(False)

        ctk.CTkButton(
            barra,
            text="Cargar imagen",
            command=self.cargar_imagen,
            height=38,
            fg_color="#d1d5db",
            hover_color="#c4c9d2",
            text_color="#111827",
        ).pack(fill="x", padx=22, pady=(44, 18))

        self.lbl_archivo = ctk.CTkLabel(barra, text="Sin imagen cargada", text_color="#4b5563", wraplength=205)
        self.lbl_archivo.pack(fill="x", padx=22, pady=(0, 24))

        ctk.CTkLabel(barra, text="Porcentaje de ruido", text_color="#111827", anchor="w").pack(fill="x", padx=22)
        self.lbl_ruido = ctk.CTkLabel(barra, text="30 %", text_color="#4b5563", anchor="w")
        self.lbl_ruido.pack(fill="x", padx=22)

        self.slider_ruido = ctk.CTkSlider(barra, from_=0, to=50, number_of_steps=50, command=self._actualizar_lbl_ruido)
        self.slider_ruido.set(30)
        self.slider_ruido.pack(fill="x", padx=22, pady=(4, 24))

        ctk.CTkLabel(barra, text="Filtro espacial 3x3", text_color="#111827", anchor="w").pack(fill="x", padx=22)
        self.selector_filtro = ctk.CTkOptionMenu(barra, values=["Media", "Mediana", "Moda"])
        self.selector_filtro.set("Mediana")
        self.selector_filtro.pack(fill="x", padx=22, pady=(6, 24))

        ctk.CTkLabel(barra, text="Radio de Fourier", text_color="#111827", anchor="w").pack(fill="x", padx=22)
        self.lbl_radio = ctk.CTkLabel(barra, text="60 px", text_color="#4b5563", anchor="w")
        self.lbl_radio.pack(fill="x", padx=22)

        self.slider_radio = ctk.CTkSlider(barra, from_=1, to=300, number_of_steps=299, command=self._actualizar_lbl_radio)
        self.slider_radio.set(60)
        self.slider_radio.pack(fill="x", padx=22, pady=(4, 28))

        ctk.CTkButton(
            barra,
            text="Procesar",
            command=self.procesar_imagen,
            height=38,
            fg_color="#16a34a",
            hover_color="#15803d",
            text_color="#ffffff",
        ).pack(fill="x", padx=22, pady=(0, 10))

        self.barra_progreso = ctk.CTkProgressBar(barra, height=10)
        self.barra_progreso.set(0)
        self.barra_progreso.pack(fill="x", padx=22, pady=(0, 6))

        self.lbl_progreso = ctk.CTkLabel(barra, text="0 %", text_color="#4b5563", anchor="center")
        self.lbl_progreso.pack(fill="x", padx=22)

        contenedor = ctk.CTkFrame(self, fg_color="#ffffff", corner_radius=0)
        contenedor.grid(row=0, column=1, sticky="nsew")
        contenedor.grid_columnconfigure((0, 1), weight=1)
        contenedor.grid_rowconfigure((0, 1), weight=1)

        self._crear_panel(contenedor, "original", "Imagen original", 0, 0)
        self._crear_panel(contenedor, "ruido", "Imagen con ruido", 0, 1)
        self._crear_panel(contenedor, "espacial", "Resultado espacial", 1, 0)
        self._crear_panel(contenedor, "frecuencia", "Resultado en campo de frecuencia", 1, 1)

    def _crear_panel(self, padre, clave, titulo, fila, columna):
        panel = ctk.CTkFrame(padre, fg_color="#ffffff", corner_radius=0)
        panel.grid(row=fila, column=columna, padx=0, pady=0, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            panel,
            text=titulo,
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#111827",
            anchor="center",
        ).grid(row=0, column=0, sticky="ew", padx=18, pady=(28, 8))

        imagen_label = ctk.CTkLabel(panel, text="Sin imagen", text_color="#9ca3af", fg_color="#ffffff", corner_radius=0)
        imagen_label.grid(row=1, column=0, padx=18, pady=(0, 24), sticky="nsew")
        self.paneles[clave] = imagen_label

    def _actualizar_lbl_ruido(self, valor):
        self.lbl_ruido.configure(text=f"{int(float(valor))} %")

    def _actualizar_lbl_radio(self, valor):
        self.lbl_radio.configure(text=f"{int(float(valor))} px")

    def _actualizar_progreso(self, valor):
        self.barra_progreso.set(valor)
        self.lbl_progreso.configure(text=f"{int(valor * 100)} %")
        self.update_idletasks()

    def _mostrar_en_panel(self, clave, imagen_pil):
        imagen_panel = preparar_imagen_panel(imagen_pil)
        imagen_ctk = ctk.CTkImage(light_image=imagen_panel, dark_image=imagen_panel, size=PANEL_IMAGE_SIZE)
        self.ctk_images[clave] = imagen_ctk
        self.paneles[clave].configure(image=imagen_ctk, text="")

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
            self.imagen_original_pil = cargar_imagen(ruta)
            self.imagen_gris = convertir_grises(self.imagen_original_pil)
            self.imagen_ruido = None
            self.resultado_espacial = None
            self.resultado_frecuencia = None

            nombre = os.path.basename(ruta)
            alto, ancho = self.imagen_gris.shape
            self.lbl_archivo.configure(text=f"{nombre}\n{ancho} x {alto} px")

            radio_maximo = max(1, min(alto, ancho) // 2)
            self.slider_radio.configure(to=radio_maximo, number_of_steps=radio_maximo)
            self.slider_radio.set(min(60, radio_maximo))
            self._actualizar_lbl_radio(self.slider_radio.get())

            self._mostrar_en_panel("original", matriz_a_imagen_pil(self.imagen_gris))
            self._limpiar_panel("ruido")
            self._limpiar_panel("espacial")
            self._limpiar_panel("frecuencia")
            self.procesar_imagen()
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{exc}")

    def _limpiar_panel(self, clave):
        self.ctk_images.pop(clave, None)
        self.paneles[clave].configure(image=None, text="Sin imagen")

    def procesar_imagen(self):
        if self.imagen_gris is None:
            messagebox.showwarning("Imagen requerida", "Primero carga una imagen.")
            return

        try:
            self.configure(cursor="watch")
            self._actualizar_progreso(0)

            ruido = int(self.slider_ruido.get())
            filtro = self.selector_filtro.get()
            radio = int(self.slider_radio.get())

            self.imagen_ruido, self.resultado_espacial, self.resultado_frecuencia = procesar_imagen(
                self.imagen_gris,
                ruido,
                filtro,
                radio,
                progreso=self._actualizar_progreso,
            )

            self._mostrar_en_panel("ruido", matriz_a_imagen_pil(self.imagen_ruido))
            self._mostrar_en_panel("espacial", matriz_a_imagen_pil(self.resultado_espacial))
            self._mostrar_en_panel("frecuencia", matriz_a_imagen_pil(self.resultado_frecuencia))
            self._actualizar_progreso(1)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo procesar la imagen:\n{exc}")
        finally:
            self.configure(cursor="")


if __name__ == "__main__":
    app = ImageSofteningApp()
    app.mainloop()
