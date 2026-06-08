import math
import csv
import random
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ===
# Q1a
# ===

def summarize(lst):
    n = len(lst)
    mean = sum(lst) / n
    var = sum((x - mean)**2 for x in lst) / (n - 1)
    return mean, var ** 0.5, n

data = [5.99342831, 4.7234714, 6.29537708, 8.04605971, 4.53169325,
        4.53172609, 8.15842563, 6.53486946, 4.06105123, 6.08512009]

mean, std, n = summarize(data)
print("1a — Mean: %.4f  Std: %.4f  N: %d" % (mean, std, n))

# =============================================
# H0: mean <= threshold   H1: mean > threshold
# =============================================

def z_test_single(values, threshold):
    mean, std, n = summarize(values)
    z = (mean - threshold) / (std / math.sqrt(n))
    p_val = 1 - stats.norm.cdf(z)
    return z, p_val

z, p = z_test_single(data, threshold=4)
print("\n1b — z=%.4f  p=%.6f  BAD=%s" % (z, p, p < 0.05))

# visualise the normal curve with p-value region shaded
x_range = np.linspace(-4, 6, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x_range, stats.norm.pdf(x_range), color="#1F497D", linewidth=2)
ax.fill_between(x_range, stats.norm.pdf(x_range),
                where=(x_range >= z), color="#C00000", alpha=0.35,
                label=f"p = {p:.6f}")
ax.axvline(z, color="#C00000", linewidth=2, linestyle="--", label=f"z = {z:.2f}")
ax.set_xlabel("z-score")
ax.set_ylabel("Probability Density")
ax.set_title("One-Sample Z-Test", fontweight="bold")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_ztest.png", dpi=150)
plt.close()

class GeneExpressionAnalyzer:

    def __init__(self, filepath, threshold=4):
        self.threshold = threshold
        self.gene_data = {}
        with open(filepath, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.patients = header[1:]
            for row in reader:
                self.gene_data[row[0]] = [float(x) for x in row[1:]]

    def summarize(self, values):
        n = len(values)
        mean = sum(values) / n
        return mean, (sum((x - mean)**2 for x in values) / (n - 1))**0.5, n

    def z_test(self, values):
        mean, std, n = self.summarize(values)
        z = (mean - self.threshold) / (std / math.sqrt(n))
        return mean, std, z, 1 - stats.norm.cdf(z)

    def run_all(self, alpha=0.05):
        results = []
        for gene, vals in self.gene_data.items():
            m, s, z, p = self.z_test(vals)
            results.append({
                "gene": gene,
                "mean": round(m, 4),
                "std": round(s, 4),
                "z_stat": round(z, 4),
                "p_value": round(p, 6),
                "label": "BAD" if p < alpha else "GOOD"
            })
        return results

# generates a demo CSV
random.seed(42)
pts = [f"Patient_{i+1}" for i in range(10)]
params = {"BRCA1":(5.5,1.8),"TP53":(5.8,1.3),"EGFR":(5.5,1.4),
          "MYC":(4.6,2.1), "KRAS":(2.5,0.9)}

with open(".venv/gene_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gene"] + pts)
    for g, (mu, sig) in params.items():
        w.writerow([g] + [round(random.gauss(mu, sig), 4) for _ in pts])

analyzer = GeneExpressionAnalyzer(".venv/gene_data.csv", threshold=4)
results = analyzer.run_all()

print("\n1c — Gene results:")
for r in results:
    print(f"  {r['gene']:<8} mean={r['mean']:>6}  p={r['p_value']:>9}  {r['label']}")

# bar chart of means with threshold line
genes = [r["gene"] for r in results]
means = [r["mean"] for r in results]
stds  = [r["std"] for r in results]
pvals = [r["p_value"] for r in results]
cols  = ["#C00000" if r["label"] == "BAD" else "#375623" for r in results]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(genes, means, color=cols, edgecolor="white", alpha=0.85, width=0.55)
ax.errorbar(genes, means, yerr=stds, fmt="none", ecolor="#333", capsize=5)
ax.axhline(4, color="#1F497D", linewidth=2, linestyle="--", label="Threshold = 4")
for i, (g, m) in enumerate(zip(genes, means)):
    ax.text(i, m + 0.08, f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Mean Expression ± Std")
ax.set_title("Mean Gene Expression Across Cohort", fontweight="bold")
ax.legend(handles=[
    mpatches.Patch(color="#C00000", alpha=0.85, label="BAD (p<0.05)"),
    mpatches.Patch(color="#375623", alpha=0.85, label="GOOD (p≥0.05)"),
    ax.get_lines()[0]
])
ax.grid(True, axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_means.png", dpi=150)
plt.close()

# p-value bar chart
fig, ax = plt.subplots(figsize=(7, 4))
lp = [-math.log10(p) if p > 0 else 10 for p in pvals]
ax.bar(genes, lp, color=cols, edgecolor="white", alpha=0.85, width=0.55)
ax.axhline(-math.log10(0.05), color="#1F497D", linewidth=2, linestyle="--", label="α = 0.05")
for i, (v, p) in enumerate(zip(lp, pvals)):
    ax.text(i, v + 0.1, f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}", ha="center", fontsize=9)
ax.set_ylabel("−log₁₀(p-value)")
ax.set_title("Z-Test P-Values by Gene", fontweight="bold")
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_pvalues.png", dpi=150)
plt.close()

# expression heatmap
matrix = np.array([analyzer.gene_data[g] for g in genes])
fig, ax = plt.subplots(figsize=(10, 3.5))
im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=9)
ax.set_xticks(range(10))
ax.set_xticklabels([f"P{i+1}" for i in range(10)])
ax.set_yticks(range(len(genes)))
ax.set_yticklabels(genes, fontweight="bold")
for i in range(len(genes)):
    for j in range(10):
        c = "white" if matrix[i, j] > 6 or matrix[i, j] < 2 else "#333"
        ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=8, color=c)
plt.colorbar(im, ax=ax, pad=0.02, label="Expression")
ax.set_title("Gene Expression Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig("plot_heatmap.png", dpi=150)
plt.close()

def scatter_genes(g1, g2, name1, name2, save_path=None):
    n  = len(g1)
    m1 = sum(g1) / n
    m2 = sum(g2) / n
    num = sum((x - m1) * (y - m2) for x, y in zip(g1, g2))
    den = (sum((x - m1)**2 for x in g1) * sum((y - m2)**2 for y in g2)) ** 0.5
    r = num / den if den else 0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(g1, g2, color="steelblue", edgecolors="#1F497D",
               s=90, alpha=0.85, linewidth=0.8, zorder=3)
    fit = np.polyfit(g1, g2, 1)
    x_line = np.linspace(min(g1) - 0.2, max(g1) + 0.2, 100)
    ax.plot(x_line, np.polyval(fit, x_line), color="#C00000",
            linewidth=1.8, linestyle="--", label=f"r = {r:.2f}")
    for i, (x, y) in enumerate(zip(g1, g2)):
        ax.annotate(f"P{i+1}", (x, y), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, color="#595959")
    ax.set_xlabel(f"{name1} Expression")
    ax.set_ylabel(f"{name2} Expression")
    ax.set_title(f"{name1} vs {name2}", fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()
    print(f"1d — Pearson r = {r:.3f}")

scatter_genes(analyzer.gene_data["BRCA1"], analyzer.gene_data["TP53"],
              "BRCA1", "TP53", save_path=".venv/plot_scatter.png")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import umap

seeds = {
    "Cancer Biology": [
        "Mutations in BRCA1 are strongly associated with hereditary breast and ovarian cancer.",
        "TP53 pathway dysregulation drives uncontrolled cell proliferation in malignant tumors.",
        "Immunotherapy targeting PD-L1 checkpoints has shown promise in solid tumor treatment.",
        "EGFR overexpression correlates with poor prognosis in non-small cell lung carcinoma.",
        "Cancer stem cells drive tumor heterogeneity and treatment resistance.",
        "Metastatic cascade involves epithelial to mesenchymal transition and matrix degradation.",
        "Oncogene amplification in MYC leads to aggressive tumor phenotypes.",
        "Apoptosis evasion through BCL2 overexpression is a hallmark of cancer.",
        "Angiogenesis driven by VEGF signaling supports tumor growth and invasion.",
        "Histopathological analysis revealed high nucleus to cytoplasm ratios indicating malignancy.",
    ],
    "Machine Learning": [
        "Convolutional neural networks learn hierarchical features from image data automatically.",
        "Gradient descent optimization minimizes the loss function during model training.",
        "Random forests aggregate many decision trees to reduce variance and overfitting.",
        "Backpropagation computes gradients by applying the chain rule through network layers.",
        "Attention mechanisms allow transformers to weight relevant tokens in sequence processing.",
        "Transfer learning leverages pretrained weights for downstream classification tasks.",
        "Cross-validation partitions data to give unbiased estimates of model generalization.",
        "Dropout regularization randomly deactivates neurons to prevent co-adaptation.",
        "Support vector machines find the maximal margin hyperplane separating class distributions.",
        "Batch normalization stabilizes training by normalizing layer activations across mini-batches.",
    ],
    "Genetics & Genomics": [
        "Single nucleotide polymorphisms represent common variation in the human genome.",
        "RNA sequencing quantifies transcriptome-wide gene expression across experimental conditions.",
        "Genome wide association studies identify loci linked to complex heritable traits.",
        "CRISPR-Cas9 enables precise genomic editing by guided endonuclease cleavage.",
        "Chromatin accessibility profiling reveals regulatory elements controlling gene transcription.",
        "Copy number variations contribute to phenotypic diversity and disease susceptibility.",
        "Epigenetic marks including methylation and acetylation modulate gene expression.",
        "Alternative splicing generates proteomic diversity from a limited number of genes.",
        "Polymerase chain reaction amplifies specific DNA sequences for downstream analysis.",
        "Transcription factor binding motifs are enriched at promoter and enhancer regions.",
    ],
    "Clinical Statistics": [
        "Kaplan-Meier curves estimate time-to-event distributions across patient cohorts.",
        "Cox proportional hazards regression models covariate effects on survival outcomes.",
        "A randomized controlled trial minimizes confounding by random treatment assignment.",
        "The p-value is the probability of results as extreme under the null hypothesis.",
        "Confidence intervals quantify uncertainty around point estimates from sample data.",
        "Bayesian inference updates prior beliefs using observed likelihood from experimental data.",
        "Multiple testing correction using Benjamini-Hochberg controls the false discovery rate.",
        "Logistic regression models binary outcomes as a function of predictor variables.",
        "Power analysis determines the sample size required to detect a given effect size.",
        "Propensity score matching balances covariates between treated and control groups.",
    ]
}

random.seed(42)
docs, labels = [], []
cats = list(seeds.keys())
for idx, (cat, sents) in enumerate(seeds.items()):
    for _ in range(80):
        docs.append(" ".join(random.choices(sents, k=random.randint(2, 4))))
        labels.append(idx)

# vectorise with TF-IDF then L2 normalise for cosine distance
tfidf = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
X_norm = normalize(tfidf.fit_transform(docs).toarray())
print(f"\n2d — TF-IDF: {X_norm.shape}")

# reduce to 2D with UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                      metric="cosine", random_state=42)
embedding = reducer.fit_transform(X_norm)

sil = silhouette_score(embedding, labels)
print(f"2d — Silhouette score: {sil:.3f}")

# UMAP scatter
la = np.array(labels)
pcols = ["#C00000", "#1F497D", "#375623", "#7B5EA7"]
pmrks = ["o", "s", "^", "D"]
fig, ax = plt.subplots(figsize=(8, 6))
for i, (cat, c, m) in enumerate(zip(cats, pcols, pmrks)):
    mask = la == i
    ax.scatter(embedding[mask, 0], embedding[mask, 1], c=c, marker=m,
               s=55, alpha=0.75, label=cat, edgecolors="white", linewidths=0.4)
ax.set_xlabel("UMAP Dim 1")
ax.set_ylabel("UMAP Dim 2")
ax.set_title(f"UMAP of Biomedical Documents  |  Silhouette = {sil:.2f}", fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.25)
plt.tight_layout()
plt.savefig("plot_umap.png", dpi=150)
plt.close()

# top TF-IDF terms per category
fn = np.array(tfidf.get_feature_names_out())
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for i, (cat, c) in enumerate(zip(cats, pcols)):
    sc = X_norm[la == i].mean(axis=0)
    top = np.argsort(sc)[-10:][::-1]
    axes[i].barh(range(10), sc[top][::-1], color=c, alpha=0.8, edgecolor="white")
    axes[i].set_yticks(range(10))
    axes[i].set_yticklabels(fn[top][::-1], fontsize=8)
    axes[i].set_title(cat, fontsize=9, fontweight="bold", color=c)
    axes[i].set_xlabel("Mean TF-IDF", fontsize=8)
    axes[i].grid(True, axis="x", linestyle="--", alpha=0.3)
plt.suptitle("Top 10 TF-IDF Terms Per Category", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plot_tfidf_terms.png", dpi=150, bbox_inches="tight")
plt.close()
print("2d — Plots saved.")