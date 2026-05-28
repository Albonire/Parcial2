#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Actividades 1, 2 y 3 - Análisis interno inicial de ESM-2.

Este script ejecuta:
1. Carga de tokenizer y modelo ESM-2.
2. Tokenización y ejecución de tres secuencias: original, mutada y alterada.
3. Reporte de tokens, IDs, formas de hidden states y atenciones.
4. Inspección básica del código fuente de Hugging Face para ESM.
"""

import inspect
import textwrap
import torch
import pandas as pd

from transformers import AutoTokenizer, EsmModel
from transformers.models.esm import modeling_esm


MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

SEQUENCES = {
    "Original": "MKTAYIAKQRQISFVKSHFSRQDILD",
    "Mutada_Y_a_F": "MKTAFIAKQRQISFVKSHFSRQDILD",
    "Alterada": "DLIDQRSFHSSKVFSIQRQKAIYATKM",
}


def load_model(model_name: str):
    """
    Carga el tokenizer y el modelo.
    Se intenta usar attn_implementation='eager' porque algunas implementaciones
    optimizadas no devuelven mapas de atención aunque output_attentions=True.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        model = EsmModel.from_pretrained(model_name, attn_implementation="eager")
        print("Modelo cargado con attn_implementation='eager'.")
    except TypeError:
        model = EsmModel.from_pretrained(model_name)
        print("Modelo cargado sin parámetro attn_implementation.")

    model.eval()
    return tokenizer, model


def analyze_sequence(name, sequence, tokenizer, model):
    """
    Tokeniza y ejecuta una secuencia con ESM-2.
    Retorna un diccionario con resultados relevantes.
    """
    inputs = tokenizer(sequence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    ids = inputs["input_ids"][0].tolist()

    attention_available = outputs.attentions is not None
    attention_shape = None
    num_attention_tensors = 0

    if attention_available:
        num_attention_tensors = len(outputs.attentions)
        attention_shape = tuple(outputs.attentions[0].shape)

    result = {
        "nombre": name,
        "secuencia": sequence,
        "longitud_sin_tokens_especiales": len(sequence),
        "tokens": tokens,
        "ids": ids,
        "input_ids_shape": tuple(inputs["input_ids"].shape),
        "last_hidden_state_shape": tuple(outputs.last_hidden_state.shape),
        "num_hidden_states": len(outputs.hidden_states) if outputs.hidden_states is not None else None,
        "attentions_disponibles": attention_available,
        "num_attention_tensors": num_attention_tensors,
        "attention_layer_0_shape": attention_shape,
        "outputs": outputs,
        "inputs": inputs,
    }

    return result


def print_model_config(model):
    """
    Imprime configuración relevante del modelo.
    """
    config = model.config
    print("\n================ CONFIGURACIÓN DEL MODELO ================")
    print(f"Modelo: {MODEL_NAME}")
    print(f"hidden_size: {config.hidden_size}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"vocab_size: {config.vocab_size}")
    print(f"max_position_embeddings: {config.max_position_embeddings}")


def inspect_component(title, obj, max_lines=45):
    """
    Muestra archivo fuente y fragmento de código de una clase/objeto.
    """
    print(f"\n================ {title} ================")

    try:
        file_path = inspect.getfile(obj)
        print(f"Archivo fuente: {file_path}")
    except Exception as exc:
        print(f"No se pudo obtener archivo fuente: {exc}")

    try:
        source = inspect.getsource(obj)
        lines = source.splitlines()
        fragment = "\n".join(lines[:max_lines])
        print("\nFragmento de código:")
        print(textwrap.indent(fragment, "    "))
        if len(lines) > max_lines:
            print(f"    ... fragmento recortado, total de líneas: {len(lines)}")
    except Exception as exc:
        print(f"No se pudo obtener código fuente: {exc}")


def main():
    print("Cargando ESM-2...")
    tokenizer, model = load_model(MODEL_NAME)

    print_model_config(model)

    print("\n================ ARQUITECTURA GENERAL ================")
    print(model)

    all_results = {}

    summary_rows = []

    for name, sequence in SEQUENCES.items():
        print(f"\n\n================ SECUENCIA: {name} ================")
        result = analyze_sequence(name, sequence, tokenizer, model)
        all_results[name] = result

        print(f"Secuencia: {sequence}")
        print(f"Longitud sin tokens especiales: {result['longitud_sin_tokens_especiales']}")
        print(f"Tokens: {result['tokens']}")
        print(f"IDs: {result['ids']}")
        print(f"input_ids shape: {result['input_ids_shape']}")
        print(f"last_hidden_state shape: {result['last_hidden_state_shape']}")
        print(f"Número de hidden_states: {result['num_hidden_states']}")
        print(f"¿Atenciones disponibles?: {result['attentions_disponibles']}")
        print(f"Número de tensores de atención: {result['num_attention_tensors']}")
        print(f"Forma atención capa 0: {result['attention_layer_0_shape']}")

        summary_rows.append({
            "tipo": name,
            "longitud": result["longitud_sin_tokens_especiales"],
            "input_ids_shape": result["input_ids_shape"],
            "last_hidden_state_shape": result["last_hidden_state_shape"],
            "num_hidden_states": result["num_hidden_states"],
            "attentions_disponibles": result["attentions_disponibles"],
            "attention_layer_0_shape": result["attention_layer_0_shape"],
        })

    print("\n\n================ TABLA RESUMEN ================")
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    print("\n\n================ INSPECCIÓN DE CÓDIGO FUENTE ================")
    print("A continuación se muestran clases principales del código fuente de ESM en Hugging Face.")

    inspect_component("Tokenizer usado por AutoTokenizer", type(tokenizer), max_lines=35)
    inspect_component("Embeddings: EsmEmbeddings", modeling_esm.EsmEmbeddings, max_lines=45)
    inspect_component("Self-Attention: EsmSelfAttention", modeling_esm.EsmSelfAttention, max_lines=60)
    inspect_component("Capa Transformer: EsmLayer", modeling_esm.EsmLayer, max_lines=50)
    inspect_component("Feed-forward intermedio: EsmIntermediate", modeling_esm.EsmIntermediate, max_lines=40)
    inspect_component("Salida feed-forward/residual: EsmOutput", modeling_esm.EsmOutput, max_lines=40)
    inspect_component("MLM Head: EsmLMHead", modeling_esm.EsmLMHead, max_lines=50)

    print("\n\nEjecución terminada correctamente.")


if __name__ == "__main__":
    main()
