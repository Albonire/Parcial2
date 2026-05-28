# Actividades 1, 2 y 3: Análisis interno inicial de ESM-2

## Actividad 1: Marco teórico

### 1. Transformer original

El Transformer es una arquitectura de aprendizaje profundo propuesta por Vaswani et al. (2017) en el artículo *Attention Is All You Need*. Su aporte principal fue reemplazar el uso de recurrencia, común en modelos como RNN o LSTM, por un mecanismo llamado atención. Esto permitió procesar secuencias de forma más paralela y eficiente.

En una tarea de secuencia, como traducir texto o analizar proteínas, el modelo debe entender relaciones entre diferentes posiciones. La idea central del Transformer es que cada elemento de la secuencia puede comparar su información con la de los demás elementos mediante self-attention. En vez de leer estrictamente de izquierda a derecha, el modelo puede calcular relaciones globales entre tokens.

En este trabajo, la arquitectura Transformer se estudia aplicada a una secuencia de aminoácidos. Es decir, cada aminoácido de una proteína se considera una unidad de entrada que el modelo transforma en representaciones numéricas.

### 2. Encoder-only, decoder-only y encoder-decoder

Existen varias familias de arquitecturas Transformer. Las tres más importantes son:

#### Encoder-only

Un modelo encoder-only procesa toda la entrada de manera bidireccional. Esto significa que cada posición puede atender tanto a tokens anteriores como posteriores. Este tipo de arquitectura se usa para comprensión, clasificación, extracción de embeddings y masked language modeling. BERT es un ejemplo clásico de esta familia (Devlin et al., 2018).

ESM-2 pertenece a esta categoría porque recibe una secuencia completa de aminoácidos y produce representaciones contextualizadas para cada posición. No está diseñado para generar una secuencia paso a paso como GPT, sino para representar y analizar secuencias proteicas completas.

#### Decoder-only

Un modelo decoder-only utiliza atención causal. Esto quiere decir que cada token solo puede mirar los tokens anteriores y a sí mismo, pero no los futuros. Esta arquitectura es común en modelos generativos como GPT, donde el objetivo es predecir el siguiente token a partir del contexto anterior (Radford et al., 2018).

GPT es útil para generación de texto porque produce tokens uno por uno. ESM-2 no funciona de esa manera, ya que su objetivo principal no es conversar ni generar texto autoregresivamente.

#### Encoder-decoder

Un modelo encoder-decoder tiene dos partes: un encoder que comprende la entrada completa y un decoder que genera una salida condicionada por esa entrada. Esta arquitectura se usa en traducción automática, resumen y tareas secuencia-a-secuencia.

El Transformer original de Vaswani et al. (2017) seguía este esquema. Sin embargo, ESM-2 se entiende mejor como un modelo encoder-only especializado en proteínas.

### 3. Tokens, embeddings y hidden states

Un modelo Transformer no procesa directamente texto, aminoácidos o imágenes en forma cruda. Primero convierte la entrada en tokens. En procesamiento de lenguaje natural, un token puede ser una palabra o subpalabra. En ESM-2, un token suele representar un aminoácido, además de tokens especiales como inicio, fin, padding o máscara.

Por ejemplo, la secuencia:

```text
MKTAYIAKQRQISFVKSHFSRQDILD
```

se convierte en una lista de tokens y luego en identificadores numéricos, llamados `input_ids`.

Después, cada identificador numérico se transforma en un vector llamado embedding inicial. El embedding inicial es una representation numérica aprendida para cada token. Sin embargo, este vector todavía no contiene todo el contexto de la secuencia.

Luego, al pasar por las capas Transformer, cada embedding se actualiza usando self-attention, capas feed-forward, conexiones residuales y normalización. El resultado de estas capas se llama hidden state o representación contextualizada.

La diferencia clave es:

- **Embedding inicial:** vector asociado al token antes de pasar por las capas Transformer.
- **Hidden state:** vector después de pasar por las capas Transformer; contiene información del token y del contexto de la secuencia.

En ESM-2, el `last_hidden_state` representa la salida final del encoder para todos los tokens de la secuencia.

### 4. Self-attention y multi-head attention

El mecanismo de self-attention permite que cada token compare su información con la de los demás tokens. Para hacerlo, el modelo crea tres proyecciones principales:

- **Q, Query o consulta:** representa qué está buscando un token.
- **K, Key o clave:** representa cómo se ofrece cada token para ser atendido.
- **V, Value o valor:** representa la información que cada token entrega si recibe atención.

La forma simplificada de la atención es:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Primero se calcula una similitud entre queries y keys. Luego se aplica `softmax` para obtener pesos de atención. Finalmente, esos pesos se usan para combinar los values.

En multi-head attention, el modelo no calcula una sola atención, sino varias en paralelo. Cada cabeza puede capturar relaciones distintas dentro de la secuencia. En proteínas, algunas cabezas podrían concentrarse en relaciones locales, mientras otras podrían distribuir atención de forma más global. Sin embargo, no se debe interpretar automáticamente una cabeza de atención como una prueba causal biológica.

### 5. Masked Language Modeling

Masked Language Modeling, o MLM, es una tarea de entrenamiento donde algunos tokens de la entrada se ocultan y el modelo debe predecirlos usando el contexto. BERT usa este objetivo para aprender representaciones bidireccionales del lenguaje (Devlin et al., 2018).

ESM-2 aplica una idea similar, pero sobre proteínas. En lugar de ocultar palabras, se pueden ocultar aminoácidos. El modelo observa los aminoácidos a ambos lados de la posición enmascarada y predice cuáles aminoácidos son más probables para esa posición.

Esto no equivale a generar una proteína como GPT genera texto. GPT genera de izquierda a derecha, token por token. ESM-2, al ser encoder-only, analiza la secuencia completa y usa contexto bidireccional.

### 6. Qué es ESM-2

ESM significa *Evolutionary Scale Modeling*. ESM-2 es un modelo de lenguaje de proteínas desarrollado por Meta AI/FAIR. Su propósito es procesar secuencias de aminoácidos mediante Transformers y producir representaciones matemáticas útiles para análisis científico.

ESM-2 no es un chatbot. No está diseñado para conversar con humanos ni para responder preguntas en lenguaje natural. Su entrada principal son secuencias de proteínas y sus salidas pueden incluir:

- tokens e identificadores de tokens,
- embeddings,
- hidden states,
- logits para masked language modeling,
- matrices de atención, si la implementación las retorna.

En ciencia de datos, ESM-2 puede usarse para comparar proteínas, analizar mutaciones, extraer embeddings para clasificadores posteriores y apoyar modelos derivados como ESMFold. ESMFold aprovecha representaciones de modelos de lenguaje de proteínas para predicción estructural, como se describe en Lin et al. (2023).

## Actividad 2: Ejecución de ESM-2 preentrenado

En esta actividad se usa el modelo:

```text
facebook/esm2_t6_8M_UR50D
```

Se analizan tres secuencias:

| Tipo | Secuencia | Descripción |
|---|---|---|
| Original | `MKTAYIAKQRQISFVKSHFSRQDILD` | Secuencia base |
| Mutada | `MKTAFIAKQRQISFVKSHFSRQDILD` | Cambio puntual Y → F |
| Alterada | `DLIDQRSFHSSKVFSIQRQKAIYATKM` | Secuencia alterada por inversión |

Para cada secuencia se reportan:

- longitud,
- tokens generados por el tokenizer,
- identificadores numéricos de tokens,
- forma de `last_hidden_state`,
- número de capas,
- número de cabezas de atención,
- forma de las atenciones, si están disponibles.

La ejecución completa se encuentra en:

```text
notebooks/esm2_actividades_1_2_3.ipynb
src/esm2_actividades_1_2_3.py
```

## Actividad 3: Inspección del código fuente

La inspección del código fuente se realiza sobre las clases de Hugging Face Transformers relacionadas con ESM. Se revisan los componentes principales del modelo y se conectan con la teoría de Transformers.

| Componente | Dónde se identifica | Qué hace |
|---|---|---|
| Tokenizer / vocabulario | `transformers.models.esm.tokenization_esm.EsmTokenizer` | Convierte aminoácidos y tokens especiales en identificadores numéricos. |
| Embeddings | `transformers.models.esm.modeling_esm.EsmEmbeddings` | Convierte `input_ids` en vectores iniciales y agrega información posicional. |
| Self-attention | `transformers.models.esm.modeling_esm.EsmSelfAttention` | Calcula Q, K, V, pesos de atención y la salida contextual de cada cabeza. |
| Multi-head attention | Dentro de `EsmSelfAttention` | Divide el espacio oculto en varias cabezas para capturar relaciones distintas en paralelo. |
| LayerNorm y residual | `EsmLayer`, `EsmOutput` | Estabilizan el entrenamiento/inferencia y preservan información entre subcapas. |
| Feed-forward | `EsmIntermediate` y `EsmOutput` | Aplica transformaciones densas por posición después de la atención. |
| MLM head | `EsmLMHead` y `EsmForMaskedLM` | Proyecta hidden states al tamaño del vocabulario para predecir tokens enmascarados. |

## Referencias

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.

Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., & Rives, A. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science, 379(6637), 1123-1130.

Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). *Improving Language Understanding by Generative Pre-Training*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS.