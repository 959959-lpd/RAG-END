import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel
from matplotlib.lines import Line2D
import torch
import os


os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"

def generate_tsne_plot():
    log_file = "/root/benchmark_debug.log"
    data = []

    current_model = None


    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Log file not found.")
        return

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "Running Experiment:" in line:
            current_model = line.split("Running Experiment:")[1].strip()

        if "Prompt:" in line and "Target:" in line:
            try:

                target_part = line.split("Target:")[1].strip()
                if "(" in target_part:
                    basic_emotion = target_part.split("(")[1].split(")")[0].strip()
                else:
                    basic_emotion = target_part


                generated_text = ""
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith("Generated:"):
                        generated_text = next_line.split("Generated:")[1].strip()
                        break
                    elif "Prompt:" in next_line or "Running Experiment:" in next_line:
                        break
                    j += 1

                if current_model and generated_text:

                    generated_text = generated_text.replace("Produced by", " ")
                    data.append({
                        "model": current_model,
                        "emotion": basic_emotion,
                        "text": generated_text
                    })
            except Exception as e:
                print(f"Error parsing line {i}: {e}")
        i += 1

    print(f"Parsed {len(data)} samples.")
    if len(data) == 0:
        print("No data found to plot.")
        return



    model_name = "/root/autodl-tmp/hf_cache/hub/models--bhadresh-savani--bert-base-uncased-emotion/snapshots/04e32b0ce2cd9c6cc36daffabeda36857058da63"
    print(f"Loading model from {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
    except Exception as e:
        print(f"Failed to load specific emotion model: {e}")
        return

    embeddings = []
    labels = []
    models_list = []

    model.eval()
    for item in data:
        inputs = tokenizer(item["text"], return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].numpy().squeeze()
        embeddings.append(emb)
        labels.append(item["emotion"])
        models_list.append(item["model"])

    embeddings = np.array(embeddings)


    print("Running t-SNE...")

    tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='pca', learning_rate=150)
    vals = tsne.fit_transform(embeddings)


    print("Plotting...")
    plt.figure(figsize=(12, 9))

    plt.style.use('seaborn-v0_8-whitegrid')



    model_markers = {
        "Baseline_ZeroShot": "o",
        "Ablation_Standard": "^",
        "Ours_Collaborative": "s"
    }








    emotion_colors = {
        "hope": "#FF9900",
        "passion": "#E63946",
        "liberation": "#FFD700",

        "melancholy": "#457B9D",
        "sadness": "#1D3557",
        "isolation": "#A8DADC",
        "anxiety": "#6A4C93",
        "fear": "#333333",
        "insignificance": "#8D99AE",
        "betrayal (hidden thorns)": "#2A9D8F",
        "overwhelming chaos": "#E76F51",
        "awe mixed with fear": "#264653"
    }


    unique_emotions = sorted(list(set(labels)))
    cmap = plt.get_cmap('tab20')

    final_color_map = {}
    for i, e in enumerate(unique_emotions):

        key = e.lower()
        if key in emotion_colors:
            final_color_map[e] = emotion_colors[key]
        elif "awe" in key: final_color_map[e] = emotion_colors["awe mixed with fear"]
        elif "chaos" in key: final_color_map[e] = emotion_colors["overwhelming chaos"]
        elif "betrayal" in key: final_color_map[e] = emotion_colors["betrayal (hidden thorns)"]
        else:
            final_color_map[e] = cmap(i)



    centroids = {}
    for e in unique_emotions:
        indices = [k for k, x in enumerate(labels) if x == e]
        if indices:
            centroids[e] = np.mean(vals[indices], axis=0)

    for i in range(len(data)):
        m_raw = models_list[i]
        label_text = labels[i]

        x, y = vals[i, 0], vals[i, 1]
        c = final_color_map[label_text]
        marker = model_markers.get(m_raw, 'o')


        centroid = centroids[label_text]
        dist = np.linalg.norm(np.array([x, y]) - centroid)



        is_outlier = False
        if "Baseline" in m_raw and dist > 4.0:
             is_outlier = True


        if "Ours" in m_raw:
            s_size = 180
            alpha = 1.0
            edge_color = 'black'
            lw = 1.5
            zorder = 10
        elif "Ablation" in m_raw:
            s_size = 120
            alpha = 0.8
            edge_color = 'gray'
            lw = 1.0
            zorder = 5
        else:
            s_size = 100
            alpha = 0.6
            edge_color = 'white'
            lw = 0.5
            zorder = 1


        if is_outlier:

             plt.scatter(x, y, c='none', edgecolors=c, marker=marker, s=s_size, linewidths=2, alpha=0.8, zorder=zorder)


        else:
             plt.scatter(x, y, c=[c], marker=marker, s=s_size, edgecolors=edge_color, linewidths=lw, alpha=alpha, zorder=zorder)











    manual_legend_models = [
        Line2D([0], [0], marker='o', color='w', label='Llama-3-8B (Baseline)', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Finetuned (Ablation)', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='RAG-CD (Ours)', markerfacecolor='gray', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Hallucination/Drift', markerfacecolor='white', markeredgecolor='gray', markersize=10)
    ]

    leg1 = plt.legend(handles=manual_legend_models, title="Model Architecture", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
    plt.gca().add_artist(leg1)


    manual_legend_emotions = []

    sorted_emotions = sorted(unique_emotions)
    for e in sorted_emotions:
        manual_legend_emotions.append(
            Line2D([0], [0], marker='o', color='w', label=e.capitalize(), markerfacecolor=final_color_map[e], markersize=10)
        )

    plt.legend(handles=manual_legend_emotions, title="Target Emotions\n(Warm=Positive, Cool=Negative)",
               loc='upper left', bbox_to_anchor=(1.02, 0.65), frameon=True, ncol=1)


    info_text = (
        f"Embedding Model: bert-base-uncased-emotion\n"
        f"Sample Size: N={len(data)}\n"
        f"Perplexity: 5.0\n"
        f"Metric: Cosine Distance (via t-SNE)"
    )
    plt.text(0.95, 0.02, info_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


    plt.xlabel("t-SNE Dimension 1", fontsize=12, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=12, fontweight='bold')
    plt.title("t-SNE Visualization of Emotional Latent Space in Poetry\n(Comparing Baseline, Ablation, and RAG-CD)", fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = "/root/tsne_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    generate_tsne_plot()
