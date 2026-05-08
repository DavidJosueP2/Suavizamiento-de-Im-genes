import numpy as np
from PIL import Image, ImageDraw


def detectar_regiones(imagen_binaria, min_area=100):
    imagen_binaria = imagen_binaria.astype(np.uint8)
    alto, ancho = imagen_binaria.shape
    blancos = imagen_binaria > 0
    negros = ~blancos
    mascara = negros if negros.sum() <= blancos.sum() else blancos
    visitado = np.zeros((alto, ancho), dtype=bool)
    regiones = []
    for fila in range(alto):
        for columna in range(ancho):
            if not mascara[fila, columna] or visitado[fila, columna]:
                continue
            pila = [(fila, columna)]
            visitado[fila, columna] = True
            min_fila = max_fila = fila
            min_columna = max_columna = columna
            area = 0
            while pila:
                af, ac = pila.pop()
                area += 1
                min_fila = min(min_fila, af)
                max_fila = max(max_fila, af)
                min_columna = min(min_columna, ac)
                max_columna = max(max_columna, ac)
                for df in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if df == 0 and dc == 0:
                            continue
                        nf, nc = af + df, ac + dc
                        if 0 <= nf < alto and 0 <= nc < ancho:
                            if not visitado[nf, nc] and mascara[nf, nc]:
                                visitado[nf, nc] = True
                                pila.append((nf, nc))
            if area >= min_area:
                regiones.append(
                    {
                        "area": int(area),
                        "x": int(min_columna),
                        "y": int(min_fila),
                        "width": int(max_columna - min_columna + 1),
                        "height": int(max_fila - min_fila + 1),
                        "centroid_x": float((min_columna + max_columna) / 2),
                        "centroid_y": float((min_fila + max_fila) / 2),
                    }
                )
    regiones.sort(key=lambda r: r["area"], reverse=True)
    for i, r in enumerate(regiones, start=1):
        r["label"] = i
    return regiones


def detectar_bounding_box(imagen_binaria, min_area=100):
    regiones = detectar_regiones(imagen_binaria, min_area)
    if not regiones:
        return None
    r = max(regiones, key=lambda x: x["area"])
    return (r["x"], r["y"], r["x"] + r["width"] - 1, r["y"] + r["height"] - 1)


def dibujar_bounding_box(imagen, bounding_box):
    if imagen.ndim == 2:
        salida = np.stack([imagen, imagen, imagen], axis=2).astype(np.uint8)
    else:
        salida = imagen.copy().astype(np.uint8)
    if bounding_box is None:
        return salida
    x1, y1, x2, y2 = bounding_box
    h, w = salida.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    color = np.array([255, 0, 0], dtype=np.uint8)
    for grosor in range(3):
        yy1 = max(0, y1 - grosor)
        yy2 = min(h - 1, y2 + grosor)
        xx1 = max(0, x1 - grosor)
        xx2 = min(w - 1, x2 + grosor)
        salida[yy1, xx1 : xx2 + 1] = color
        salida[yy2, xx1 : xx2 + 1] = color
        salida[yy1 : yy2 + 1, xx1] = color
        salida[yy1 : yy2 + 1, xx2] = color
    return salida


def dibujar_regiones_numeradas(imagen, regiones):
    base = dibujar_bounding_box(imagen, None)
    pil = Image.fromarray(base)
    draw = ImageDraw.Draw(pil)
    for r in regiones:
        x, y = r["x"], r["y"]
        w, h = r["width"], r["height"]
        x2, y2 = x + w - 1, y + h - 1
        etiqueta = str(r["label"])
        for offset in range(3):
            draw.rectangle(
                (x - offset, y - offset, x2 + offset, y2 + offset), outline=(0, 180, 70)
            )
        tx, ty = x, max(0, y - 22)
        draw.rectangle((tx, ty, tx + 26, ty + 18), fill=(255, 255, 255))
        draw.text((tx + 6, ty + 2), etiqueta, fill=(0, 90, 180))
    return np.array(pil, dtype=np.uint8)


def recortar_regiones(imagen, regiones, margen=8):
    if imagen.ndim == 2:
        base = np.stack([imagen, imagen, imagen], axis=2).astype(np.uint8)
    else:
        base = imagen.astype(np.uint8)
    h, w = base.shape[:2]
    recortes = []
    for r in regiones:
        x1 = max(0, r["x"] - margen)
        y1 = max(0, r["y"] - margen)
        x2 = min(w, r["x"] + r["width"] + margen)
        y2 = min(h, r["y"] + r["height"] + margen)
        recorte = base[y1:y2, x1:x2].copy()
        recortes.append({"region": r, "imagen": recorte})
    return recortes
