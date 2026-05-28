# Actividades 4, 5, 6 y 7: Representaciones internas, atención, MLM y usos de ESM-2

## Actividad 4: Visualización de embeddings y hidden states

En esta actividad se extraen representaciones internas de ESM-2 a partir del `last_hidden_state`. Cada token de la secuencia recibe un vector. Estos vectores se llaman hidden states o representaciones contextualizadas, porque incorporan información del contexto completo de la secuencia.

Se construyen los siguientes elementos:

1. **Embedding por residuo:** cada aminoácido tiene su propio vector dentro del `last_hidden_state`.
2. **Embedding global de proteína:** se calcula promediando los vectores de los residuos, excluyendo tokens especiales. Esto produce un único vector que representa toda la proteína.
3. **Similitud coseno:** se compara la similitud entre la secuencia original, la mutada y la alterada usando sus embeddings globales.
4. **Proyección 2D (PCA):** se reducen los embeddings a dos dimensiones para visualizarlos en un plano.

### Interpretación esperada

- La similitud entre la secuencia original y la mutada suele ser alta, porque difieren en un solo aminoácido.
- La similitud entre la secuencia original y la alterada suele ser menor, porque el orden cambió considerablemente.
- La visualización PCA no demuestra significado biológico por sí sola; es una representación matemática reducida.

## Actividad 5: Matrices de atención

Las matrices de atención muestran cuánto atiende cada posición de la secuencia a las demás posiciones, para una capa y una cabeza específicas. La forma esperada de las atenciones es:

```text
(batch, num_heads, sequence_length, sequence_length)
```

Cada matriz es de tamaño L x L, donde L es el número de tokens. Cada fila representa la distribución de atención de un token hacia los demás.

Se presentan, como mínimo:

- una matriz de atención de una capa temprana,
- una matriz de atención de una capa profunda,
- dos cabezas distintas de una misma capa,
- una comparación entre la secuencia original y la mutada.

### Limitación importante

La atención muestra patrones de relación estadística, pero no demuestra causalidad biológica. No se debe afirmar que una cabeza de atención explica un mecanismo biológico real.

### Si las atenciones no aparecen

Algunas implementaciones optimizadas no retornan mapas de atención. Si esto ocurre, se documenta el intento, se explica técnicamente la causa y se centra el análisis en hidden states y logits.

## Actividad 6: Masked Language Modeling

En esta actividad se oculta un aminoácido de la secuencia usando el token de máscara del tokenizer. Luego se usa `EsmForMaskedLM` para obtener los aminoácidos más probables en esa posición.

Se reporta:

- qué posición fue enmascarada,
- qué aminoácidos recibieron mayor puntaje (top-k),
- si el modelo predijo el aminoácido original.

### Por qué esto no es generación tipo GPT

GPT genera texto token por token de izquierda a derecha. ESM-2, al ser encoder-only, observa el contexto completo a ambos lados de la posición enmascarada. Por eso predice una posición concreta usando contexto bidireccional, en lugar de generar una secuencia paso a paso.

## Actividad 7: Usos reales y limitaciones

### Usos reales

1. Comparación de proteínas mediante embeddings.
2. Análisis de mutaciones.
3. Apoyo a predicción de estructura mediante modelos derivados como ESMFold.
4. Anotación funcional o clasificación posterior.
5. Búsqueda semántica de proteínas en bases vectoriales.
6. Priorización de variantes para ingeniería de proteínas.

### Limitaciones

- No descubre medicamentos automáticamente.
- No reemplaza experimentos de laboratorio.
- La atención no implica causalidad biológica.
- Tiene sesgos de los datos de entrenamiento.
- Requiere recursos de cómputo según el tamaño del modelo.

## Referencias

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.

Lin, Z. et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science, 379(6637), 1123-1130.

Radford, A. et al. (2018). *Improving Language Understanding by Generative Pre-Training*.

Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
