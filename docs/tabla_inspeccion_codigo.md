# Tabla de inspección del código fuente de ESM-2

| Componente | Clase / archivo esperado en Hugging Face | Qué hace | Relación con la teoría |
|---|---|---|---|
| Tokenizer / vocabulario | `EsmTokenizer`, `tokenization_esm.py` | Convierte letras de aminoácidos y tokens especiales en IDs numéricos. | Es la etapa que transforma la secuencia simbólica en entrada discreta para el Transformer. |
| Embeddings | `EsmEmbeddings`, `modeling_esm.py` | Convierte IDs en vectores iniciales y agrega posiciones. | Corresponde al embedding inicial antes de la contextualización. |
| Self-attention | `EsmSelfAttention`, `modeling_esm.py` | Calcula Query, Key, Value y pesos de atención. | Implementa la fórmula de atención del Transformer. |
| Multi-head attention | `EsmSelfAttention`, métodos internos como `transpose_for_scores` | Divide la atención en varias cabezas. | Permite aprender diferentes tipos de relaciones entre residuos. |
| LayerNorm y residual | `EsmLayer`, `EsmOutput`, `LayerNorm` | Normaliza y conserva información entre subcapas. | Ayuda a estabilizar el flujo de información por las capas. |
| Feed-forward | `EsmIntermediate`, `EsmOutput` | Aplica una MLP por posición. | Transforma cada representación contextual después de la atención. |
| MLM head | `EsmLMHead`, `EsmForMaskedLM` | Produce logits sobre el vocabulario para predecir tokens ocultos. | Se usa para masked language modeling. |