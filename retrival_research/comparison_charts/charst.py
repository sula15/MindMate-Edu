import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preparation
data = [
    ["Spacy", "HNSW", "L2", 0.015, 0.85, 12.5, 2.5],
    ["Spacy", "HNSW", "COSINE", 0.1608, 0.9164, 13.21, None],
    ["BERT", "HNSW", "L2", 0.0130, 0.8578, 658.08, 2.32],
    ["BERT", "HNSW", "COSINE", 0.0142, 0.8578, 657.18, 3],
    ["BERT", "HNSW", "IP", 0.0142, 0.8578, 657.18, 3],
    ["BERT", "IVF_FLAT", "L2", 0.0200, 0.85, 700, 3.5],
    ["BERT", "IVF_FLAT", "COSINE", 0.0202, 0.8575, 701, 3.6],
    ["BERT", "IVF_FLAT", "IP", 0.0205, 0.8555, 699, 3.7],
    ["BERT", "SCANN", "L2", 0.010, 0.87, 250, 2.8],
    ["BERT", "SCANN", "COSINE", 0.0120, 0.875, 255, 3.0],
    ["BERT", "SCANN", "IP", 0.0130, 0.865, 260, 3.2],
    ["SciBERT", "HNSW", "L2", 0.0140, 0.87, 755.27, 2.035],
    ["SciBERT", "HNSW", "COSINE", 0.015, 0.91, 755.5, 2.1],
    ["SciBERT", "HNSW", "IP", 0.0155, 0.92, 755, 2.2],
    ["SciBERT", "IVF_FLAT", "L2", 0.025, 0.87, 700, 4],
    ["SciBERT", "IVF_FLAT", "COSINE", 0.026, 0.875, 701, 4.1],
    ["SciBERT", "IVF_FLAT", "IP", 0.027, 0.865, 700, 4.2],
    ["SciBERT", "SCANN", "L2", 0.009, 0.89, 270, 3.0],
    ["SciBERT", "SCANN", "COSINE", 0.010, 0.895, 275, 3.2],
    ["SciBERT", "SCANN", "IP", 0.011, 0.88, 280, 3.5],
    ["CLIP", "HNSW", "L2", 0.015, 0.88, 35, 2.5],
    ["CLIP", "HNSW", "COSINE", 0.01, 0.9333, 0.02, 2],
    ["CLIP", "HNSW", "IP", 0.012, 0.91, 40, 2.5],
    ["CLIP", "IVF_FLAT", "L2", 0.0200, 0.89, 100, 3.5],
    ["CLIP", "IVF_FLAT", "COSINE", 0.0220, 0.88, 100, 3.6],
    ["CLIP", "IVF_FLAT", "IP", 0.023, 0.87, 100, 3.7],
    ["CLIP", "SCANN", "L2", 0.010, 0.89, 80, 3],
    ["CLIP", "SCANN", "COSINE", 0.012, 0.9, 85, 3.2],
    ["CLIP", "SCANN", "IP", 0.014, 0.88, 90, 3.4],
]

columns = ["Model", "Indexing", "Search", "Search Speed (s)", "Accuracy", "Memory Usage (MB)", "Index Build Time (s)"]
df = pd.DataFrame(data, columns=columns)

# Data Visualization
sns.set(style="whitegrid")

# Plot Accuracy Comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Model", y="Accuracy", hue="Indexing")
plt.title("Accuracy Comparison by Model and Indexing Type")
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.legend(title="Indexing Type")
plt.show()

# Plot Search Speed Comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Model", y="Search Speed (s)", hue="Indexing")
plt.title("Search Speed Comparison by Model and Indexing Type")
plt.ylabel("Search Speed (seconds)")
plt.xlabel("Model")
plt.legend(title="Indexing Type")
plt.show()

# Plot Memory Usage Comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Model", y="Memory Usage (MB)", hue="Indexing")
plt.title("Memory Usage Comparison by Model and Indexing Type")
plt.ylabel("Memory Usage (MB)")
plt.xlabel("Model")
plt.legend(title="Indexing Type")
plt.show()

# Plot Index Build Time Comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Model", y="Index Build Time (s)", hue="Indexing")
plt.title("Index Build Time Comparison by Model and Indexing Type")
plt.ylabel("Index Build Time (seconds)")
plt.xlabel("Model")
plt.legend(title="Indexing Type")
plt.show()

