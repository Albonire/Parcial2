# Análisis interno de ESM-2

Trabajo de Segundo Corte - Ciencia de Datos  
Universidad de Pamplona - Ingeniería de Sistemas

## Tema

Este repositorio contiene las primeras tres actividades del trabajo:

1. **Marco teórico:** Transformers, ESM-2, tokens, embeddings, self-attention y masked language modeling.
2. **Ejecución de ESM-2 preentrenado:** uso del modelo `facebook/esm2_t6_8M_UR50D` con secuencia original, mutada y alterada.
3. **Inspección del código fuente:** identificación de tokenizer, embeddings, self-attention, multi-head attention, LayerNorm, feed-forward y MLM head.

## Abrir notebook en Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Albonire/Parcial2/blob/main/notebooks/esm2_actividades_1_2_3.ipynb)

Enlace directo:

```text
https://colab.research.google.com/github/Albonire/Parcial2/blob/main/notebooks/esm2_actividades_1_2_3.ipynb
```

## Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── esm2_actividades_1_2_3.ipynb
├── informe/
│   └── actividad_1_2_3.md
├── src/
│   └── esm2_actividades_1_2_3.py
└── docs/
    └── tabla_inspeccion_codigo.md
```

## Ejecución local en Fedora 44

Crear entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Ejecutar script local:

```bash
python src/esm2_actividades_1_2_3.py
```

O abrir el notebook:

```bash
jupyter notebook notebooks/esm2_actividades_1_2_3.ipynb
```

Si no tienes Jupyter:

```bash
pip install notebook
```

## Modelo usado

Se usa el modelo recomendado por la guía:

```text
facebook/esm2_t6_8M_UR50D
```

Este modelo tiene aproximadamente 8 millones de parámetros y permite analizar tokens, hidden states y, si la configuración lo permite, mapas de atención.

## Secuencias usadas

| Tipo | Secuencia | Descripción |
|---|---|---|
| Original | `MKTAYIAKQRQISFVKSHFSRQDILD` | Secuencia base de ejemplo |
| Mutada | `MKTAFIAKQRQISFVKSHFSRQDILD` | Mutación puntual Y → F |
| Alterada | `DLIDQRSFHSSKVFSIQRQKAIYATKM` | Secuencia alterada/invertida parcialmente |

## Autor
**Anderson González**  
GitHub: [Albonire](https://github.com/Albonire)

## Nota importante

Este trabajo no entrena ESM-2, no hace fine-tuning y no crea un chatbot.  
El objetivo es analizar internamente un transformer científico preentrenado aplicado a secuencias de proteínas.
