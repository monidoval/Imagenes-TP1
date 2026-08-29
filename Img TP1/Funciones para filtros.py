import numpy as np
from PIL import Image


'''def tomar_valores_vecindad(matriz, radio, x, y):

    return matriz[x - radio : x + radio + 1, y - radio : y + radio + 1]'''

def aplicar_filtro_media(imagen, tam_fil):
    arr_img = np.array(imagen)
    filas, col = arr_img.shape
    img_filtrada = arr_img.copy()
    radio = tam_fil // 2   # valor entero de la división

    total_filas = (filas - radio) - radio
    contador_filas = 0

    print(f'Inicio filtrado de media directa ({tam_fil}x{tam_fil})...')

    for x in range(radio, filas - radio):
        for y in range(radio, col - radio):
            # Vecindad obtenida por slicing directo (se puede cambiar por tomar valores vecindad)
            vecindad = arr_img[x - radio : x + radio + 1,
                               y - radio : y + radio + 1]
            
            # Promedio directo con NumPy
            nuevo_valor = np.mean(vecindad)
            img_filtrada[x, y] = int(nuevo_valor)

        contador_filas += 1
        porcentaje = (contador_filas / total_filas) * 100
        print(f'\rProgreso: {porcentaje:.2f}%', end="")

    print('\nFiltrado completado.')
    return Image.fromarray(img_filtrada)

def aplicar_filtro_mediana(imagen, tam_filtro):
    arr_imagen = np.array(imagen)
    filas, col = arr_imagen.shape
    img_filtrada = arr_imagen.copy()
    radio = tam_filtro // 2   # parte entera de la división

    total_filas = (filas - radio) - radio
    contador_filas = 0

    print(f'Inicio de filtrado mediana ({tam_filtro}x{tam_filtro})')

    for x in range(radio, filas - radio):
        for y in range(radio, col - radio):
            # Vecindad obtenida por slicing directo
            vecindad = arr_imagen[x - radio : x + radio + 1,
                                  y - radio : y + radio + 1]
            
            # Mediana directa con NumPy
            nuevo_valor = np.median(vecindad)
            img_filtrada[x, y] = int(nuevo_valor)

        contador_filas += 1
        porcentaje = (contador_filas / total_filas) * 100
        print(f'\rProgreso: {porcentaje:.2f}%', end="")

    print('\nFiltrado completado.')
    return Image.fromarray(img_filtrada)

def aplicar_filtro_mediana_ponderada(imagen, repeticiones):
    arr_imagen = np.array(imagen)
    filas, col = arr_imagen.shape
    img_filtrada = arr_imagen.copy()

    # El tamaño del filtro se deduce de la longitud de repeticiones
    tam_lado = int(np.sqrt(len(repeticiones)))
    radio = tam_lado // 2  

    total_filas = (filas - radio) - radio
    contador_filas = 0

    print(f'Inicio de filtrado mediana ponderada ({tam_lado}x{tam_lado})')

    for x in range(radio, filas - radio):
        for y in range(radio, col - radio):
            # Vecindad obtenida por slicing directo
            vecindad = arr_imagen[x - radio : x + radio + 1,
                                  y - radio : y + radio + 1].flatten()

            # Aplicar ponderación repitiendo cada valor según su peso
            vecindad_rep = np.repeat(vecindad, repeticiones)

            # Calcular la mediana ponderada directamente
            nuevo_valor = np.median(vecindad_rep)
            img_filtrada[x, y] = int(nuevo_valor)

        contador_filas += 1
        porcentaje = (contador_filas / total_filas) * 100
        print(f'\rProgreso: {porcentaje:.2f}%', end="")

    print('\nFiltrado completado.')
    return Image.fromarray(img_filtrada)

def aplicar_filtro_gauss(imagen, desviacion):
    arr_imagen = np.array(imagen)

    # Tamaño del filtro en función de sigma
    k = round(2 * desviacion + 1)
    if k % 2 == 0:
        k += 1
    radio = k // 2   # más simple y correcto para filtros impares

    filas, col = arr_imagen.shape
    img_filtrada = arr_imagen.copy()

    total_filas = (filas - radio) - radio
    contador_filas = 0

    print(f'Inicio de filtrado Gaussiano ({k}x{k}, Sigma={desviacion})')

    # Precalcular la cuadrícula de coordenadas relativas
    x_coords, y_coords = np.mgrid[-radio:radio+1, -radio:radio+1]
    exponentes = -(x_coords**2 + y_coords**2) / (2 * desviacion**2)
    kernel = (1 / (2 * np.pi * desviacion**2)) * np.exp(exponentes)
    kernel = kernel / np.sum(kernel)  # normalización

    for x in range(radio, filas - radio):
        for y in range(radio, col - radio):
            # Vecindad obtenida por slicing directo
            vecindad = arr_imagen[x - radio : x + radio + 1,
                                  y - radio : y + radio + 1]

            # Aplicar el kernel Gaussiano
            nuevo_valor = np.sum(vecindad * kernel)
            img_filtrada[x, y] = int(nuevo_valor)

        contador_filas += 1
        porcentaje = (contador_filas / total_filas) * 100
        print(f'\rProgreso: {porcentaje:.2f}%', end="")

    print('\nFiltrado completado.')
    return 

def aplicar_filtro_realce_directo(imagen, tam_filtro):
    arr_img = np.array(imagen)
    filas, col = arr_img.shape
    img_filtrada = arr_img.copy()
    radio = tam_filtro // 2   # más simple y correcto para filtros impares

    # Construcción del kernel de realce
    kernel = -np.ones((tam_filtro, tam_filtro), dtype=int)
    kernel[radio, radio] = (tam_filtro**2) - 1

    total_filas = (filas - radio) - radio
    contador_filas = 0

    print(f'Inicio filtrado de realce ({tam_filtro}x{tam_filtro})...')

    for x in range(radio, filas - radio):
        for y in range(radio, col - radio):
            # Vecindad obtenida por slicing directo
            vecindad = arr_img[x - radio : x + radio + 1,
                               y - radio : y + radio + 1]

            # Aplicar el kernel de realce
            nuevo_valor = np.sum(vecindad * kernel)
            img_filtrada[x, y] = np.clip(nuevo_valor, 0, 255)

        contador_filas += 1
        porcentaje = (contador_filas / total_filas) * 100
        print(f'\rProgreso: {porcentaje:.2f}%', end="")

    print('\nFiltrado completado.')
    return Image.fromarray(img_filtrada)