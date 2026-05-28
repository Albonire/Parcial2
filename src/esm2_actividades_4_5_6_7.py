#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Actividades 4, 5, 6 y 7 - Análisis interno de ESM-2.

Este script ejecuta:
4. Visualización de embeddings y hidden states (similitud coseno y PCA).
5. Matrices de atención por capa y por cabeza.
6. Masked Language Modeling con top-k.
7. Usos reales y limitaciones (resumen impreso).

Las figuras se guardan en la carpeta outputs/.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Permite guardar figuras sin entorno gráfico
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM


MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

SEQUENCES = {
    "Original": "MKTAYIAKQRQISFVKSHFSRQDILD",
    "Mutada_Y_a_F": "MKTAFIAKQRQISFVKSHFSRQDILD",
    "Alterada": "DLIDQRSFHSSKVFSIQRQKAIYATKM",
}

OUTPUT_DIR = "outputs"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_models(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        model = EsmModel.from_pretrained(model_name, attn_implementation="eager")
        print("EsmModel cargado con attn_implementation='eager'.")
    except TypeError:
        model = EsmModel.from_pretrained(model_name)
        print("EsmModel cargado sin attn_implementation.")

    model.eval()

    mlm_model = EsmForMaskedLM.from_pretrained(model_name)
    mlm_model.eval()

    return tokenizer, model, mlm_model


def run_model(sequence, tokenizer, model):
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
    return inputs, outputs


def get_global_embedding(outputs):
    """
    Promedio de hidden states sin tokens especiales (primero y ultimo).
    """
    last_hidden = outputs.last_hidden_state[0]  # (seq_len, hidden)
    # Excluir token inicial y token final
    residue_states = last_hidden[1:-1, :]
    global_emb = residue_states.mean(dim=0).numpy()
    return global_emb, residue_states.numpy()


def activity_4(tokenizer, model):
    print("\n" + "="*80)
    print("ACTIVIDAD 4: Embeddings y hidden states")
    print("="*80)

    global_embeddings = {}
    residue_embeddings = {}

    for name, seq in SEQUENCES.items():
        inputs, outputs = run_model(seq, tokenizer, model)
        g_emb, r_emb = get_global_embedding(outputs)
        global_embeddings[name] = g_emb
        residue_embeddings[name] = r_emb

        print(f"\nSecuencia: {name}")
        print(f"last_hidden_state shape: {tuple(outputs.last_hidden_state.shape)}")
        print(f"Embedding por residuo shape: {r_emb.shape}")
        print(f"Embedding global shape: {g_emb.shape}")

    # Similitud coseno entre embeddings globales
    names = list(global_embeddings.keys())
    matrix = np.array([global_embeddings[n] for n in names])
    sim = cosine_similarity(matrix)

    df_sim = pd.DataFrame(sim, index=names, columns=names)
    print("\nSimilitud coseno entre embeddings globales:")
    print(df_sim.to_string())

    df_sim.to_csv(os.path.join(OUTPUT_DIR, "similitud_coseno.csv"))

    # PCA de embeddings por residuo combinados
    all_vectors = []
    all_labels = []
    for name in names:
        vecs = residue_embeddings[name]
        all_vectors.append(vecs)
        all_labels.extend([name] * vecs.shape[0])

    all_vectors = np.vstack(all_vectors)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_vectors)

    plt.figure(figsize=(8, 6))
    for name in names:
        idx = [i for i, lab in enumerate(all_labels) if lab == name]
        plt.scatter(coords[idx, 0], coords[idx, 1], label=name, alpha=0.7)
    plt.title("PCA de embeddings por residuo")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend()
    plt.tight_layout()
    pca_path = os.path.join(OUTPUT_DIR, "pca_embeddings.png")
    plt.savefig(pca_path, dpi=150)
    plt.close()
    print(f"\nFigura PCA guardada en: {pca_path}")


def plot_attention(att_matrix, tokens, title, filename):
    plt.figure(figsize=(8, 7))
    plt.imshow(att_matrix, aspect="auto")
    plt.colorbar(label="peso de atención")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Figura de atención guardada en: {path}")


def activity_5(tokenizer, model):
    print("\n" + "="*80)
    print("ACTIVIDAD 5: Matrices de atención")
    print("="*80)

    seq = SEQUENCES["Original"]
    inputs, outputs = run_model(seq, tokenizer, model)

    if outputs.attentions is None:
        print("Las atenciones no están disponibles en esta configuración.")
        print("Se recomienda revisar attn_implementation o documentar el intento.")
        return

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    num_layers = len(outputs.attentions)
    num_heads = outputs.attentions[0].shape[1]

    print(f"Número de capas con atención: {num_layers}")
    print(f"Número de cabezas por capa: {num_heads}")

    # Capa temprana
    early_layer = 0
    att_early = outputs.attentions[early_layer][0, 0].detach().cpu().numpy()
    plot_attention(att_early, tokens,
                   f"Atención - capa {early_layer}, cabeza 0",
                   "atencion_capa_temprana.png")

    # Capa profunda
    deep_layer = num_layers - 1
    att_deep = outputs.attentions[deep_layer][0, 0].detach().cpu().numpy()
    plot_attention(att_deep, tokens,
                   f"Atención - capa {deep_layer}, cabeza 0",
                   "atencion_capa_profunda.png")

    # Dos cabezas distintas de la misma capa
    if num_heads >= 2:
        att_h0 = outputs.attentions[early_layer][0, 0].detach().cpu().numpy()
        att_h1 = outputs.attentions[early_layer][0, 1].detach().cpu().numpy()
        plot_attention(att_h0, tokens,
                       f"Atención - capa {early_layer}, cabeza 0",
                       "atencion_capa0_cabeza0.png")
        plot_attention(att_h1, tokens,
                       f"Atención - capa {early_layer}, cabeza 1",
                       "atencion_capa0_cabeza1.png")

    # Comparación original vs mutada en la misma capa/cabeza
    seq_mut = SEQUENCES["Mutada_Y_a_F"]
    inputs_mut, outputs_mut = run_model(seq_mut, tokenizer, model)
    if outputs_mut.attentions is not None:
        tokens_mut = tokenizer.convert_ids_to_tokens(inputs_mut["input_ids"][0])
        att_mut = outputs_mut.attentions[early_layer][0, 0].detach().cpu().numpy()
        plot_attention(att_mut, tokens_mut,
                       f"Atención mutada - capa {early_layer}, cabeza 0",
                       "atencion_mutada_capa0_cabeza0.png")


def activity_6(tokenizer, mlm_model):
    print("\n" + "="*80)
    print("ACTIVIDAD 6: Masked Language Modeling")
    print("="*80)

    base_seq = "MKTAYIAKQRQISFVKSHFSRQDILD"
    mask_token = tokenizer.mask_token

    # Enmascarar la primera Y
    masked_sequence = base_seq.replace("Y", mask_token, 1)
    print(f"Secuencia original: {base_seq}")
    print(f"Secuencia enmascarada: {masked_sequence}")

    masked_inputs = tokenizer(masked_sequence, return_tensors="pt")
    mask_index = (masked_inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        mlm_outputs = mlm_model(**masked_inputs)

    logits = mlm_outputs.logits[0, mask_index, :]
    topk = torch.topk(logits, k=5, dim=-1)

    print("\nTop-5 tokens predichos para la posición enmascarada:")
    for score, token_id in zip(topk.values[0], topk.indices[0]):
        token = tokenizer.convert_ids_to_tokens([token_id.item()])[0]
        print(f"  {token}  ->  score: {float(score):.4f}")

    print("\nNota: el aminoácido original en esa posición era 'Y'.")
    print("Esto no es generación tipo GPT; ESM-2 usa contexto bidireccional.")


def activity_7():
    print("\n" + "="*80)
    print("ACTIVIDAD 7: Usos reales y limitaciones")
    print("="*80)

    usos = [
        "Comparación de proteínas mediante embeddings.",
        "Análisis de mutaciones.",
        "Apoyo a predicción de estructura (ESMFold).",
        "Anotación funcional o clasificación posterior.",
        "Búsqueda semántica de proteínas en bases vectoriales.",
        "Priorización de variantes para ingeniería de proteínas.",
    ]

    limitaciones = [
        "No descubre medicamentos automáticamente.",
        "No reemplaza experimentos de laboratorio.",
        "La atención no prueba causalidad biológica.",
        "Tiene sesgos de los datos de entrenamiento.",
        "Requiere recursos de cómputo según el tamaño del modelo.",
    ]

    print("\nUsos reales:")
    for u in usos:
        print(f"  - {u}")

    print("\nLimitaciones:")
    for l in limitaciones:
        print(f"  - {l}")


def main():
    ensure_output_dir()
    print("Cargando modelos ESM-2...")
    tokenizer, model, mlm_model = load_models(MODEL_NAME)

    activity_4(tokenizer, model)
    activity_5(tokenizer, model)
    activity_6(tokenizer, mlm_model)
    activity_7()

    print("\n\nEjecución terminada. Revisa la carpeta outputs/ para las figuras.")


if __name__ == "__main__":
    main()
