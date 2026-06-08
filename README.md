# Biomedical Data Analysis

A small, from scratch toolkit for statistical analysis, classification, and
unsupervised exploration of biomedical data. It covers hypothesis testing, an
object oriented gene expression pipeline, correlation analysis, document
embedding with TF-IDF and UMAP, and an image based malignancy measurement.

## What it does

- **Statistics from scratch.** Sample mean and standard deviation with Bessel's
  correction, and a one sample z test with p values via `scipy.stats`.
- **Object oriented analysis pipeline.** `GeneExpressionAnalyzer` loads a
  gene by sample CSV and runs the z test across every gene, returning a clean
  GOOD / BAD classification report.
- **Correlation and visualization.** Pearson r computed by hand, plus scatter
  plots, error bar charts, and an expression heatmap.
- **NLP and representation learning.** A full document embedding pipeline:
  TF-IDF vectorization, L2 normalization, UMAP reduction to 2D, and a
  silhouette score to quantify how well topics separate.
- **Image analysis.** Nucleus to cytoplasm ratio estimated by pixel level color
  thresholding, with notes on where a U-Net would be more robust.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python main.py
```

## Outputs

Running the script generates these figures:

- `plot_ztest.png` — normal curve with the p value region shaded
- `plot_means.png` — mean expression per gene with significance coloring
- `plot_pvalues.png` — negative log p values per gene
- `plot_heatmap.png` — expression heatmap across patients
- `plot_scatter.png` — BRCA1 vs TP53 correlation
- `plot_umap.png` — UMAP projection of biomedical documents
- `plot_tfidf_terms.png` — top TF-IDF terms per category

## Techniques used

Python, NumPy, scikit-learn (TF-IDF, silhouette), UMAP, SciPy, Matplotlib,
object oriented design, hypothesis testing, correlation analysis,
dimensionality reduction, and basic computer vision.
