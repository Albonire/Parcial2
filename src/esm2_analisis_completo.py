
import os
import inspect
import textwrap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM
from transformers.models.esm import modeling_esm

# Configuration
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
SEQUENCES = {
    "Original": "MKTAYIAKQRQISFVKSHFSRQDILD",
    "Mutada":   "MKTAFIAKQRQISFVKSHFSRQDILD",
    "Alterada": "DLIDQRSFHSSKVFSIQRQKAIYATKM",
}

OUTPUT_DIR = "/home/fabian/Documents/university/CienciasDeDatoss/Parcial2/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Try to load with eager implementation for explicit attentions
    try:
        model = EsmModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
        print("EsmModel loaded with attn_implementation='eager'.")
    except Exception:
        model = EsmModel.from_pretrained(MODEL_NAME)
        print("EsmModel loaded with default implementation.")
    model.eval()

    mlm_model = EsmForMaskedLM.from_pretrained(MODEL_NAME)
    mlm_model.eval()

    results = {}
    print("\n--- Running Inference ---")
    for name, seq in SEQUENCES.items():
        inputs = tokenizer(seq, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, output_attentions=True)
        
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        ids = inputs["input_ids"][0].tolist()
        
        print(f"Processed {name}: length={len(seq)}, tokens={len(tokens)}")
        results[name] = (inputs, out, tokens, ids)

    # 1. Global Similiarity
    print("\n--- Calculating Similarity ---")
    glob_embs = {}
    for name, (inputs, out, tokens, ids) in results.items():
        # Average pooling excluding CLS and EOS tokens
        emb = out.last_hidden_state[0, 1:-1, :].mean(dim=0).numpy()
        glob_embs[name] = emb

    names = list(glob_embs.keys())
    matrix = np.array([glob_embs[n] for n in names])
    sim = cosine_similarity(matrix)
    df_sim = pd.DataFrame(sim, index=names, columns=names)
    df_sim.to_csv(os.path.join(OUTPUT_DIR, "similitud_coseno.csv"))
    print(df_sim)

    # 2. PCA
    print("\n--- Generating PCA ---")
    all_vecs, all_labels = [], []
    for name in names:
        _, out, _, _ = results[name]
        v = out.last_hidden_state[0, 1:-1, :].numpy()
        all_vecs.append(v)
        all_labels += [name] * v.shape[0]
    
    all_vecs = np.vstack(all_vecs)
    coords = PCA(n_components=2).fit_transform(all_vecs)

    plt.figure(figsize=(10, 7))
    for name in names:
        idx = [i for i, l in enumerate(all_labels) if l == name]
        plt.scatter(coords[idx, 0], coords[idx, 1], label=name, alpha=0.6, edgecolors='w')
    plt.title("PCA de embeddings por residuo (ESM-2)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, \"pca_unificado.png\"), dpi=150)\n",
    plt.close()


    # 3. Attention Maps
    print("\n--- Plotting Attentions ---")
    def plot_att(att, tokens, title, fname):
        plt.figure(figsize=(10, 9))
        plt.imshow(att, cmap='viridis')
        plt.title(title)
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.colorbar(label="Attention weight")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
        plt.close()

    _, out_o, tokens_o, _ = results["Original"]
    if out_o.attentions is not None:
        n_layers = len(out_o.attentions)
        # Layer 0, Head 0
        plot_att(out_o.attentions[0][0, 0].numpy(), tokens_o, 
                 "Atención - Capa 0, Cabeza 0 (Secuencia Original)", "atencion_capa_0_cabeza_0.png")
        # Deep Layer
        plot_att(out_o.attentions[-1][0, 0].numpy(), tokens_o, 
                 f"Atención - Capa {n_layers-1}, Cabeza 0", "atencion_capa_profunda.png")
        
        # Compare Original vs Mutated
        _, out_m, tokens_m, _ = results["Mutada"]
        plot_att(out_m.attentions[0][0, 0].numpy(), tokens_m, 
                 "Atención - Capa 0, Cabeza 0 (Secuencia Mutada)", "atencion_mutada_capa_0_cabeza_0.png")

    # 4. Masked Language Modeling
    print("\n--- Masked Language Modeling ---")
    orig_seq = SEQUENCES["Original"]
    # Masking 'Y' at index 4 (0-indexed)
    masked_seq = list(orig_seq)
    masked_seq[4] = tokenizer.mask_token
    masked_seq_str = "".join(masked_seq)
    
    inputs_mlm = tokenizer(masked_seq_str, return_tensors="pt")
    mask_idx = (inputs_mlm.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        logits = mlm_model(**inputs_mlm).logits

    probs = torch.softmax(logits[0, mask_idx, :], dim=-1)
    top5 = torch.topk(probs, 5)
    
    mlm_results = []
    print(f"Predictions for masked position in {orig_seq} (was 'Y'):")
    for s, i in zip(top5.values[0], top5.indices[0]):
        tok = tokenizer.convert_ids_to_tokens([i.item()])[0]
        print(f"  {tok}: {float(s):.4f}")
        mlm_results.append({"token": tok, "prob": float(s)})
    
    pd.DataFrame(mlm_results).to_csv(os.path.join(OUTPUT_DIR, "mlm_predictions.csv"), index=False)

    print(f"\nAll artifacts generated in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
