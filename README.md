# 🧭 pose_CAD_registration

Este repositorio implementa un sistema de registro de pose para modelos CAD 3D a partir de imágenes 2D, basado principalmente en el algoritmo propuesto por **Atoche (2011)**. Se emplean correspondencias punto-rayo y visibilidad geométrica para estimar la rotación y traslación que alinean el modelo con la vista de una cámara calibrada.

---

## 🎯 Objetivo

Desarrollar una herramienta funcional de visión por computadora que:

- Detecte bordes relevantes en la imagen (SUSAN, Canny).
- Genere rayos proyectados desde puntos de borde utilizando la matriz intrínseca de cámara.
- Empareje estos rayos con puntos visibles del modelo CAD.
- Calcule la transformación óptima (pose) que alinee el modelo con la imagen.

---

## 🛠️ Herramientas y métodos utilizados

- **Lenguaje**: Python 3.x
- **Librerías**: Open3D (con CUDA), OpenCV, NumPy, Matplotlib
- **Métodos clave**:
  - Correspondencia punto-rayo (CLP)
  - Estimación robusta de pose (SVD + Huber)
  - Visibilidad por geometría (normales perpendiculares)
  - Transformaciones iterativas y multivista

---

## 📂 Desglose por scripts

### `intento_viejo.py` — 🌀 Primer intento de optimización

> Un algoritmo funcional que intenta minimizar el error entre rayos proyectados y puntos CAD.

🎬 **Video**:  
[![Rotaciones en tiempo real](https://img.youtube.com/vi/fMsy3DXZB2s/0.jpg)](https://youtu.be/fMsy3DXZB2s?si=36iL8rra4udyjm5_)


📌 **Resultado**:
- Calcula error ponderado y muestra convergencia.
- 🚨 Se atasca en **mínimos locales**.  
- No escapa por falta de exploración angular.

  ![image](https://github.com/user-attachments/assets/c8a937c4-0461-49cb-a069-5fe2329662f9)


---

### `visible_modern.py` — 🧰 Visualizador moderno

> Visualizador interactivo que permite inspeccionar:

- Las **listas geométricas**: vértices, aristas, normales, puntos interpolados.
- La imagen texturizada.
- Rayos proyectados desde la imagen.
- La **cámara calibrada** con perspectiva.

🎯 **Función clave**: Diagnóstico geométrico completo.

  ![image](https://github.com/user-attachments/assets/a4b26420-cd9c-47cf-bd84-b128e65139d7)


---

### `visible_rotations.py` — 🔄 Transformaciones en tiempo real

> Aplica rotaciones dinámicas al modelo CAD y **recalcula visibilidad**.

📍 Ideal para comprobar que el sistema responde correctamente a transformaciones.

 
![image5](https://github.com/user-attachments/assets/56cefa01-82c6-4229-ba3f-ab3bf2d5e0db)



---

### `32_optimizacionn.py` — 🔬 Optimización según Atoche

> Implementación casi completa del método de **Atoche (2011)**:
- Construcción de listas geométricas (Listas 1–4).
- Evaluación de visibilidad (Lista 5) mediante normales perpendiculares.
- Emparejamiento rayo ↔ modelo.
- Estimación de pose con pesos robustos y SVD.
- Iteración controlada por error ponderado.

🎬 **Video**:  
[![Optimización Atoche](https://img.youtube.com/vi/jx56KG5L3dE/0.jpg)](https://youtu.be/jx56KG5L3dE?si=6sKcNj0x9pLJSd_T)

  ![image](https://github.com/user-attachments/assets/b9f580b7-017a-4391-8813-386540cbb26a)


📌 **Resultado**:
- Logro parcial: estructura funcional y visualización completa.
- 🚨 **Fracaso** en converger a la pose real debido a **mínimos locales persistentes**.
- ⚠️ **Alta complejidad del modelo Benchy** contribuyó a errores y lentitud de procesamiento.

---

## 🚧 Problemas encontrados

| Problema                         | Descripción                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| ❌ Mínimos locales               | La optimización converge en poses incorrectas.                              |
| 🐌 Modelo CAD complejo (Benchy) | Alto número de vértices dificultó la visibilidad y emparejamiento.         |
| ⏱️ Tiempo limitado               | No se integraron estrategias de exploración global ni depuración fina.     |

---

## 🧠 Conclusión

Este proyecto demostró la **complejidad real** de aplicar algoritmos teóricos de visión por computadora en escenarios prácticos. Replicar la metodología de Atoche permitió:

- Comprender en profundidad el registro CAD↔imagen.
- Identificar los cuellos de botella geométricos y algorítmicos.
- Visualizar en tiempo real los efectos de transformaciones y correspondencias.

Aunque el sistema **no logró estimar correctamente la pose**, dejó como legado una plataforma sólida para exploraciones futuras que incorporen:

- Métodos de escape de mínimos locales.
- Reducción de complejidad geométrica.
- Integración con redes neuronales o aprendizaje supervisado.

---

## 📚 Referencias

- Atoche, A. (2011). *Pose Estimation of 3D CAD Models from Monocular Images* (Tesis de Maestría).
- Wunsch, P. (1996). *Registration of CAD-models to images by iterative inverse perspective matching*.
- [Open3D](http://www.open3d.org/)
- [OpenCV](https://opencv.org/)
