# Usos reales y limitaciones de ESM-2

## Usos reales

| Uso | Descripción | Cómo se relaciona con este trabajo |
|---|---|---|
| Comparación de proteínas | Convertir proteínas en embeddings y medir similitud coseno. | Se aplica en la Actividad 4 comparando secuencias. |
| Análisis de mutaciones | Comparar embeddings o logits entre secuencia original y mutada. | Se observa el cambio entre original y mutada. |
| Apoyo a predicción de estructura | Modelos como ESMFold usan representaciones de ESM-2. | Se menciona como aplicación derivada. |
| Anotación funcional / clasificación posterior | Usar embeddings como entrada de clasificadores. | Se explica el flujo conceptual, sin entrenar clasificador. |
| Búsqueda semántica de proteínas | Indexar embeddings en bases vectoriales para buscar proteínas similares. | Relacionado con la idea de similitud entre vectores. |
| Priorización de variantes | Ordenar mutaciones candidatas para ingeniería de proteínas. | Relacionado con análisis de mutaciones. |

## Limitaciones

- ESM-2 no descubre medicamentos automáticamente.
- ESM-2 no reemplaza experimentos de laboratorio.
- Las matrices de atención no prueban causalidad biológica.
- El modelo refleja sesgos de los datos de entrenamiento.
- Modelos grandes requieren recursos de cómputo significativos.
- Las predicciones son estadísticas; requieren validación científica.
