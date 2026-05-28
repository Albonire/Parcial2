#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis interno de ESM-2: PROYECTO COMPLETO (Actividades 1-7)

Este script integra todas las fases del análisis:
1. Inspección de arquitectura y formas.
2. Extracción de hidden states y embeddings globales.
3. Visualización de similitud coseno y PCA.
4. Generación de mapas de atención.
5. Experimento de Masked Language Modeling.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM

# CONFIGURACIÓN
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
SEQUENCES = {
    "Original": "MKTAYIAKQRQISFVKSHFSRQDILD",
    "Mutada": "MKTAFIAKQRQISFVKSHFSRQDILD",
    "Alterada": "DLIDQRSFHSSKVFSIQRQKAIYATKM",
}
OUTPUT_DIR = "outputs"

def setup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Cargando modelos: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:
        model = EsmModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    except:
        model = EsmModel.from_pretrained(MODEL_NAME)
    model.eval()
    mlm_model = EsmForMaskedLM.from_pretrained(MODEL_NAME)
    mlm_model.eval()
    return tokenizer, model, mlm_model

def run_analysis(tokenizer, model, mlm_model):
    print("\n--- 1. Extracción de Hidden States ---")
    results = {}
    for name, seq in SEQUENCES.items():
        inputs = tokenizer(seq, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, output_attentions=True)
        results[name] = (inputs, out)
        print(f"Secuencia '{name}' procesada. Shape: {out.last_hidden_state.shape}")

    print("\n--- 2. Similitud y PCA ---")
    glob_embs = {}
    for name, (inputs, out) in results.items():
        # Media de residuos (excluyendo tokens especiales)
        glob_embs[name] = out.last_hidden_state[0, 1:-1, :].mean(dim=0).numpy()
    
    names = list(glob_embs.keys())
    matrix = np.array([glob_embs[n] for n in names])
    sim = cosine_similarity(matrix)
    df_sim = pd.DataFrame(sim, index=names, columns=names)
    print("Matriz de similitud coseno:")
    print(df_sim)
    df_sim.to_csv(os.path.join(OUTPUT_DIR, "similitud_unificada.csv"))

    print("\n--- 3. Mapas de Atención ---")
    orig_out = results["Original"][1]
    if orig_out.attentions:
        att = orig_out.attentions[0][0, 0].detach().numpy()
        plt.imshow(att, aspect="auto")
        plt.colorbar()
        plt.title("Atención: Capa 0, Cabeza 0 (Original)")
        plt.savefig(os.path.join(OUTPUT_DIR, "atencion_unificada.png"))
        print("Gráfica de atención guardada.")

    print("\n--- 4. Masked Language Modeling ---")
    masked_seq = SEQUENCES["Original"].replace("Y", tokenizer.mask_token, 1)
    inputs_m = tokenizer(masked_seq, return_tensors="pt")
    mask_idx = (inputs_m.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    with torch.no_grad():
        logits = mlm_model(**inputs_m).logits
    top5 = torch.topk(logits[0, mask_idx, :], 5)
    print(f"Predicción para posición enmascarada (era 'Y'):")
    for s, i in zip(top5.values[0], top5.indices[0]):
        token = tokenizer.convert_ids_to_tokens([i.item()])[0]
        print(f"  Token: {token} | Score: {s:.4f}")

def main():
    tokenizer, model, mlm_model = setup()
    run_analysis(tokenizer, model, mlm_model)
    print("\nAnálisis completo finalizado. Resultados en la carpeta 'outputs/'.")

if __name__ == "__main__":
    main()
