# Análisis interno de ESM-2 (Proyecto Unificado)

Trabajo de Segundo Corte - Ciencia de Datos  
Universidad de Pamplona - Ingeniería de Sistemas

Este repositorio contiene el análisis científico integral del modelo de lenguaje de proteínas ESM-2. Se exploran desde los conceptos teóricos hasta la extracción de representaciones matemáticas y visualizaciones de atención.

## Acceso Directo e Informes

Toda la lógica de ejecución y visualización está consolidada en un solo notebook interactivo:

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Albonire/Parcial2/blob/main/notebooks/esm2_analisis_completo.ipynb)

### Otros Documentos
- 📄 [Informe Técnico (HTML)](https://github.com/Albonire/Parcial2/blob/main/informe/informe_tecnico.html)
- 📊 [Diapositivas del Proyecto](https://github.com/Albonire/Parcial2/blob/main/notebooks/diapositivas_esm2.html)
- 📗 [Trabajo Escrito (PDF)](https://github.com/Albonire/Parcial2/blob/main/trabajo_segundo_corte_esm2_ciencia_datos_unipamplona_2026_1.pdf)

## Estructura del Proyecto Unificado

```text
.
├── README.md               (Esta guía)
├── requirements.txt        (Dependencias del proyecto)
├── notebooks/
│   └── esm2_analisis_completo.ipynb   # Análisis interactivo total
├── src/
│   └── esm2_analisis_completo.py      # Script ejecutable unificado
├── informe/
│   └── informe_tecnico.html           # Reporte final diseñado (ver en navegador)
└── docs/
    ├── tabla_inspeccion_codigo.md     # Componentes del Transformer
    └── usos_reales_y_limitaciones.md  # Análisis de aplicaciones
```

## Guía de Uso

### 1. Generación de Resultados
Para ejecutar el análisis completo localmente y generar las tablas y gráficas:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/esm2_analisis_completo.py
```

### 2. Informe Final
El archivo `informe/informe_tecnico.html` es un documento profesional autocontenido. Para obtener el PDF:
1. Abre el archivo en tu navegador (Chrome/Firefox).
2. Presiona `Ctrl + P`.
3. Selecciona **Guardar como PDF**.
4. **IMPORTANTE:** En "Más configuraciones", activa **Gráficos de fondo** para preservar el diseño de la portada y las cajas de texto.

## Autor
**Anderson González**  
GitHub: [@Albonire](https://github.com/Albonire)
