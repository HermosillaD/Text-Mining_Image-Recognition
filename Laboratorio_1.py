#!/usr/bin/env python
# coding: utf-8

# # Hoja de Trabajo # 1
# ## _Darling Hermosilla_ | carné No. 22006414

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# #### Problema 1:
# Desarrolle una función la cual reciba dos parámetros, una imagen y un entero llamado color, la función debe devolver una imagen la cual tenga activos los canales de color según los siguientes puntos:
# - Si el parámetro color vale 1, la imagen debe mostrar activos únicamente el color azul.
# - Si el parámetro color vale 2, la imagen debe mostrar activos únicamente el color verde.
# - Si el parámetro color vale 3, la imagen debe mostrar activos únicamente el color rojo.
# - Si el parámetro color vale 10, la imagen debe mostrar activos únicamente los colores rojo y verde.
# - Si el parámetro color vale 20, la imagen debe mostrar activos únicamente los colores verde y azul.
# - Si el parámetro color vale 30, la imagen debe mostrar activos únicamente los colores azul y rojo.

# In[ ]:


img = cv2.imread("Atitlan_2.PNG")


# In[ ]:


img.shape


# In[ ]:


plt.imshow(img)
plt.show()


# In[ ]:


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()


# In[ ]:


#dimensiones
alto = img.shape[0]
ancho = img.shape[1]
lienzo = np.zeros((alto, ancho, 3))


# In[ ]:


for i in range(0, alto):
        for j in range(0, ancho):
            pixel = img[i, j]
        
            blue = pixel[2]
            green = pixel[1]
            red = pixel[0]
            
            def prueba(num):
                if prueba == 2:
                    lienzo[i, j] = [blue, 0, 0]
                    cv2.imwrite("azul.jpg", lienzo)
                    azul = cv2.imread("azul.jpg")
                    azul = cv2.cvtColor(azul, cv2.COLOR_BGR2RGB)
                    plt.imshow(azul)
                    plt.show()
                elif prueba == 1:
                    lienzo[i, j] = [0, green, 0]
                    cv2.imwrite("verde.jpg", lienzo)
                    verde = cv2.imread("verde.jpg")
                    verde = cv2.cvtColor(verde, cv2.COLOR_BGR2RGB)
                    plt.imshow(verde)
                    plt.show()
                else:
                    lienzo[i, j] = [0, 0, red]
                    cv2.imwrite("roja.jpg", lienzo)
                    roja = cv2.imread("roja.jpg")
                    roja = cv2.cvtColor(roja, cv2.COLOR_BGR2RGB)
                    plt.imshow(roja)
                    plt.show()


# In[ ]:


prueba(1)


# #### Problema 2:
# En el .zip del laboratorio se le compartió un conjunto de imágenes en escala de grises (imagen1, imagen2, perro) estas imágenes fueron creadas utilizando una escala de grises en 3D, cree una función que dadas las 3 imagenes se construya la imagen original a color.

# In[ ]:


img1= cv2.imread('perro_salida_gray_azul.jpg')
img2= cv2.imread('perro_salida_gray_verde.jpg')
img3= cv2.imread('perro_salida_gray_rojo.jpg')

#Reescalamiento
res1= cv2.resize(img1,(300,300),interpolation= cv2.INTER_CUBIC)
res2= cv2.resize(img2,(300,300),interpolation= cv2.INTER_CUBIC)
res3= cv2.resize(img3,(300,300),interpolation= cv2.INTER_CUBIC)

#SUMA
suma_parte1= cv2.add(res1,res2)
suma_final= cv2.add(suma_parte1,res3)

cv2.imshow('imagen final suma',suma_final)
#cv2.destroyAllwindows()
cv2.waitKey(0)


# #### Problema 3:
# Cree una función que dada una imagen cree una escala de grises en tres dimensiones, tome en cuenta que su función debe crear 3 imágenes como salida. Para entregar este ejercicio debe incluir una las imágenes que haya utilizado como prueba y el resultado de las misma, no puede utilizar la imagen del Problema #2.

# In[ ]:


from PIL import Image

img = Image.open("Foto.jpg")

r, g, b = img.split()

nula = r.point(lambda x: 0)

roja = Image.merge("RGB", (r, nula, nula))
verde = Image.merge("RGB", (nula, g, nula))
azul = Image.merge("RGB", (nula, nula, b))


# #### Problema 4:
# Cree una función que dada una imagen, muestre el histograma de cada canal de color y el de escala de grises (utilice un promedio aritmético para su escala de grises, no puede usar funciones de opencv), sus histogramas deben incluir una l+inea vertical la cual muestre el valor de la media de la distribución.

# In[ ]:


hist_np2 = np.bincount(img.ravel(), minlength=256)

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.plot(range(256), hist_cv, 'r')
plt.subplot(223), plt.plot(range(256), hist_np, 'b')
plt.subplot(224), plt.plot(hist_np2)
plt.show()


# #### Problema 5:
# Investigue en que consiste el enfoque de escala de grises ponderado, luego de esto implemente una función que dada una imagen, realice una escala de grises ponderada (notar que no existe una solución única).

# In[ ]:


img = cv2.imread('foto.jpg')
alto = img.shape[0]
ancho = img.shape[1]


# In[ ]:


gris_ponderado = np.zeros((alto, ancho, 1))


# In[ ]:


for i in range(0, alto):
    for j in range(0, ancho):
        pixel = img[i, j]
        
        blue = pixel[2]
        green = pixel[1]
        red = pixel[0]
        
        #escala de grises ponderada
        gris_ponderado[i, j] = int(0.299*blue + 0.587*green + 0.114*red) 


# In[ ]:


cv2.imwrite("gris_ponderado.jpg", gris_ponderado)
monocromo = cv2.imread("gris_ponderado.jpg")
monocromo = cv2.cvtColor(monocromo, cv2.COLOR_BGR2RGB)
plt.imshow(monocromo)
plt.show()


# #### Problema 6:
# Investigue brevemente en que consiste el espacio de color HSV y como se mapean colores a dicho espacio, para entregar este ejercicio puede hacerlo por medio de Markdown en el mismo Notebook donde trabajó los demás ejercicios.

# El modelo HSV (del inglés Hue, Saturation, Value – Matiz, Saturación, Valor), también llamado HSB (Hue, Saturation, Brightness – Matiz, Saturación, Brillo), define un modelo de color en términos de sus componentes.
# Fue creado en 1978 por Alvy Ray Smith. Se trata de una transformación no lineal del espacio de color RGB, y se puede usar en progresiones de color.
# 
# En la matiz HSV se representa por una región circular; una región triangular separada, puede ser usada para representar la saturación y el valor del color. Normalmente, el eje horizontal del triángulo denota la saturación, mientras que el eje vertical corresponde al valor del color. De este modo, un color puede ser elegido al tomar primero el matiz de una región circular, y después seleccionar la saturación y el valor del color deseados de la región triangular. Se representa como un grado de ángulo cuyos valores posibles van de 0 a 360° (aunque para algunas aplicaciones se normalizan del 0 al 100%). Cada valor corresponde a un color. Ejemplos: 0 es rojo, 60 es amarillo y 120 es verde.
# 
# De forma intuitiva se puede realizar la siguiente transformación para conocer los valores básicos RGB:
# Disponemos de 360 grados dónde se dividen los 3 colores RGB, eso da un total de 120º por color, sabiendo esto podemos recordar que el 0 es rojo RGB(1, 0, 0), 120 es verde RGB(0, 1, 0) y 240 es azul RGB(0, 0, 1). Para colores mixtos se utilizan los grados intermedios, el amarillo, RGB(1, 1, 0) está entre rojo y verde, por lo tanto 60º. Se puede observar como se sigue la secuencia de sumar 60 grados y añadir un 1 o quitar el anterior:
# Cono del modelo HSV.
# 
# - 0º = RGB(1, 0, 0)
# - 60º = RGB(1, 1, 0)
# - 120º = RGB(0, 1, 0)
# - 180º = RGB(0, 1, 1)
# - 240º = RGB(0, 0, 1)
# - 300º = RGB(1, 0, 1)
# - 360º = 0º

# In[ ]:


rueda = cv2.imread('Rueda.jpg')
cono = cv2.imread("Cono.jpg")


# In[ ]:


rueda = cv2.cvtColor(rueda, cv2.COLOR_BGR2RGB)
plt.imshow(rueda)
plt.show()


# In[ ]:


cono = cv2.cvtColor(cono, cv2.COLOR_BGR2RGB)
plt.imshow(cono)
plt.show()


# ## Fin
