# Urban PM2.5 Imbalance Evaluation

Open, reproducible research materials supporting the article:

**Rethinking Model Evaluation for Urban PM2.5 Classification:
Imbalance, Temporal Validation and Computational Cost**
Case study: Lisbon (2021–2023), using official data from the European Environment Agency (EEA).

## 🔗 Project links

- 🌐 **Project page (methodology & evaluation framework)**  
  https://jcaceres-academic.github.io/urban-pm25-imbalance-evaluation/

- 📘 **Reproducible notebook**  
  https://jcaceres-academic.github.io/urban-pm25-imbalance-evaluation/notebook.html

## 📁 Repository overview

| Folder | Description |
|--------|------------|
| `docs/` | Rendered project website and reproducible notebooks (Quarto / GitHub Pages) |
| `scripts/` | Full reproducible Python pipeline (data preparation, modelling, evaluation, visualisation) |
| `figures/` | Publication-ready high-resolution figures (300 dpi, submission-ready formats) |
| `images/` | PNG visual assets used in the rendered HTML pages (web-optimised versions of figures) |
| `data/` | Consolidated parquet datasets used in the analysis (EEA PM2.5 data, daily aggregation, classification outputs) |

## 🔁 Research focus

This project implements a reproducible evaluation framework for daily urban PM2.5 classification under real-world deployment constraints.
The study emphasises:
<ul>
  <li>Structural class imbalance (rare high-pollution events)</li>
  <li>Strict temporal validation (Train: 2021–2022 | Test: 2023)</li>
  <li>Balanced Accuracy and Macro-F1 as primary metrics</li>
  <li>Explicit integration of computational cost (training + inference time)</li>
</ul>
Rather than proposing a new algorithm, the repository rethinks how predictive models should be compared when operational deployment and rare-event detection matter.


## 📜 License

Creative Commons Attribution 4.0 (CC BY 4.0)

---

➡️ For the full project description, workflow, references, and educational context,  
see the **project website** linked above.

