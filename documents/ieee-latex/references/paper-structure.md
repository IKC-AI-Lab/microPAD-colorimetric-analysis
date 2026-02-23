# IEEE Paper Structure Reference

## Table of Contents

1. [Document Preamble](#document-preamble)
2. [Title and Author Block](#title-and-author-block)
3. [Abstract and Keywords](#abstract-and-keywords)
4. [Introduction (11-Point Flow)](#introduction)
5. [Materials and Methods](#materials-and-methods)
6. [Results and Discussion](#results-and-discussion)
7. [Conclusion](#conclusion)
8. [Acknowledgements and Bibliography](#acknowledgements-and-bibliography)
9. [Standard Tables](#standard-tables)
10. [Standard Figures](#standard-figures)
11. [Equations](#equations)
12. [Quality Checklist](#quality-checklist)

---

## Document Preamble

```latex
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage[pdftex]{graphicx}
\usepackage[cmex10]{amsmath}
\usepackage{amsfonts}
\usepackage{multirow}
\usepackage{array}
\usepackage[lofdepth,lotdepth]{subfig}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{makecell}
\usepackage{hyperref}
\graphicspath{ {./figures/} }
```

For journal articles, change to: `\documentclass[journal]{IEEEtran}`

## Title and Author Block

**Title pattern**: "[Technology]-Based [AI Method] for [Task] of [Analytes] with [Device Type] in [Sample Matrix]"

```latex
% With known authors:
\author{
\IEEEauthorblockN{First1 Last1\IEEEauthorrefmark{1},
First2 Last2\IEEEauthorrefmark{1},
First3 Last3\IEEEauthorrefmark{2}*}
\IEEEauthorblockA{\IEEEauthorrefmark{1}Department of Electrical and Electronics Engineering,\\
Izmir Katip Celebi University, Izmir, Turkey}
\IEEEauthorblockA{\IEEEauthorrefmark{2}Department of Biomedical Engineering,\\
Izmir Katip Celebi University, Izmir, Turkey\\
Email: corresponding.author@ikcu.edu.tr}
}

% Without authors (use TODO):
% TODO-USER: Provide author names and emails
% Format: 'FirstName LastName <email>; FirstName2 LastName2 <email2>'
```

**Common departments at IKCU**:
- Electrical and Electronics Engineering Graduate Program
- Biomedical Engineering Graduate Program
- Department of Electrical and Electronics Engineering
- Department of Biomedical Engineering

**Last author** is typically corresponding author (marked with *).

## Abstract and Keywords

**Abstract** (~150-200 words, single paragraph, 6-sentence structure):

1. Context: importance of noninvasive/accessible sensing
2. Problem: limitations of current methods
3. Approach: "In this study, [method] was developed..."
4. Implementation: AI model, app name, analytes
5. Results: LOD values, accuracy, processing time
6. Impact: "the integrated system holds great promise for point-of-care testing..."

**Keywords** (5-7 terms, always include):
- smartphone, colorimetric, microfluidic paper-based analytical device (uPAD)
- Add: specific ML/DL method, target analytes, sample type, application domain

```latex
\begin{IEEEkeywords}
Smartphone, colorimetric, microfluidic paper-based analytical device, deep learning, [analyte], point-of-care testing.
\end{IEEEkeywords}
```

## Introduction

Follow this 11-point flow:

1. **Opening**: "Recently, [noninvasive/smartphone-based] methods have emerged as crucial..."
2. **Problem**: Limitations of invasive measurements (infection risk, difficulty for chronic patients)
3. **Alternatives**: Body fluids (sweat, saliva, tears, urine) as noninvasive sources
4. **Detection**: Colorimetric advantages (simplicity, visual determination, resource-limited settings)
5. **uPAD integration**: "Colorimetric detection can be easily integrated into uPADs which offer simple, low-cost, portable diagnostic applications"
6. **Challenge**: Color interpretation affected by camera optics and ambient light
7. **AI solution**: "Recently, AI has been successfully applied to interpret color changes robustly"
8. **ML vs. DL**: ML requires less processing power but DL handles complex problems better
9. **Classification vs. Regression**: Distinguish the approaches and their tradeoffs
10. **Cloud limitations**: "Cloud-based systems require continuous server operation... data transfer delays"
11. **This work**: "Here, [contribution]..." - state novelty clearly

## Materials and Methods

### 2.1 Materials
- List chemicals with purity and supplier: "Chemical name (purity %) (Supplier, Country)"
- Include: Whatman qualitative filter paper grade 1
- Sample preparation references (artificial saliva/sweat/tears)

### 2.2 Design and Fabrication of uPADs
- Wax printing: Xerox ColorQube 8900, Microsoft PowerPoint design
- Heating: "180 C for 120-180 s"
- Hydrophobic barrier formation
- Detection zone modification: enzyme mixtures, chromogenic agents
- Volumes: typically 0.8-1 uL droplets

### 2.3 Image Acquisition
- Dataset creation rationale
- 7 lighting conditions: H, F, S, HF, HS, FS, HFS (Halogen 2700K, Fluorescent 4000K, Sunlight 6500K)
- 5 angles: 30, 60, 90, 120, 150 degrees
- Multi-phone validation: typically 4 brands (2 iOS, 2 Android)
- Total images = concentrations x lightings x angles x phones x replicates
- Train/validation/test: 80:20, with 20% of training for validation

### 2.4 AI-Based Classification/Regression
- Model architecture details (layer-by-layer for DL)
- Feature extraction (for ML): RGB, HSV, L*a*b*, YUV color spaces; texture features
- Training: optimizer (Adam), loss function, epochs, batch size, learning rate
- Hyperparameter optimization: grid search

### 2.5 Smartphone Application: [AppName]
- Android Studio, TensorFlow Lite
- Model conversion: HDF5 (.h5) to .tflite
- Firebase (cloud) or offline capability
- User interface description

### 2.6 Selectivity (if applicable)
- Interfering species tested, concentration levels

## Results and Discussion

### 3.1 Sensor Performance
- Visual color change description, concentration-dependent intensity
- LOD calculation and values, dynamic range

### 3.2 Model Comparison (Tables Essential)
- Multiple ML/DL models compared
- Best model identification with bold formatting
- Confusion matrix figure, ROC-AUC curves

### 3.3 Smartphone Application Demo
- Multi-panel screenshots (6-9 panels)
- Processing workflow: home, gallery/camera, crop, ROI, upload, results

### 3.4 Real Sample Testing
- Volunteer testing (with consent), comparison with physiological ranges

### 3.5 Selectivity Results
- Bar chart: target vs. interferents

### 3.6 State-of-the-Art Comparison
- Table comparing with recent literature, emphasize superior performance

## Conclusion

- Restate contribution in past tense
- Summarize key results (accuracy, LOD, processing time)
- "To the best of our knowledge, this is the first study..." (if applicable)
- Advantages: "easy-to-use operation, rapid response, low-cost, high selectivity, consistent repeatability"
- Applications: "point-of-care testing, nonlaboratory settings, sports medicine, self-health monitoring"

## Acknowledgements and Bibliography

```latex
\section*{ACKNOWLEDGEMENTS}
This research was supported by the Scientific and Technological Research Council of Turkey
(TUBITAK) (project no. XXXXXX) and by the scientific research projects coordination unit
of Izmir Katip Celebi University (project no. XXXX-XXX-XXXX-XXXX).

\bibliographystyle{IEEEtran}
\bibliography{references}
```

Or use inline `\begin{thebibliography}{00}...\end{thebibliography}`.

**Author Contributions** (if multiple authors):
```latex
\section*{Author Contributions}
First1 Last1: Methodology (equal); Validation (equal); Writing--original draft (equal).
First2 Last2: Software (lead); Investigation (equal); Writing--original draft (equal).
Last Author: Conceptualization; Supervision; Funding acquisition (lead); Writing--review \& editing.
```

## Standard Tables

### Table 1: Smartphone Camera Properties
```latex
\begin{table}[t]
\caption{Smartphone Camera Properties}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Smartphone} & \textbf{Resolution} & \textbf{Optics} & \textbf{MP} \\
\midrule
iPhone X     & 4032$\times$3024 & f/1.8 & 12 \\
iPhone 11    & 4032$\times$3024 & f/1.8 & 12 \\
Samsung A75  & 4000$\times$3000 & f/1.8 & 48 \\
Realme C55   & 4000$\times$3000 & f/1.8 & 64 \\
\bottomrule
\end{tabular}
\label{tab:smartphones}
\end{table}
```

### Table 2: ML/DL Model Comparison
```latex
\begin{table}[t]
\caption{Performance Comparison of ML/DL Models}
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{MSE} & \textbf{MAE} & \textbf{R$^2$} & \textbf{RMSE} \\
\midrule
Linear Regression  & [XX] & [XX] & [XX] & [XX] \\
Random Forest      & [XX] & [XX] & [XX] & [XX] \\
\textbf{Proposed}  & \textbf{[XX]} & \textbf{[XX]} & \textbf{[XX]} & \textbf{[XX]} \\
\bottomrule
\end{tabular}
\label{tab:model_comparison}
\end{table}
```

### Table 3: Per-Class Performance
```latex
\begin{table}[t]
\caption{Classification Performance per Concentration Class}
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\
\midrule
0 mM    & [XX] & [XX] & [XX] & [XX] \\
0.1 mM  & [XX] & [XX] & [XX] & [XX] \\
Average & [XX] & [XX] & [XX] & --   \\
\bottomrule
\end{tabular}
\label{tab:per_class}
\end{table}
```

### Table 4: State-of-the-Art Comparison
```latex
\begin{table}[t]
\caption{Comparison with State-of-the-Art Methods}
\centering
\begin{tabular}{llccc}
\toprule
\textbf{Ref.} & \textbf{Sample} & \textbf{Method} & \textbf{Metric} & \textbf{LOD} \\
\midrule
{[1]}          & Blood   & ML  & [XX]\% & [XX] $\mu$M \\
\textbf{Ours}  & \textbf{Saliva} & \textbf{DL} & \textbf{[XX]\%} & \textbf{[XX] $\mu$M} \\
\bottomrule
\end{tabular}
\label{tab:sota}
\end{table}
```

## Standard Figures

### Figure 1: System Overview
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/system_overview}
\caption{Schematic illustration of the AI-based colorimetric quantification of [analytes]
in [sample]. Color change in the detection zones was imaged under various combinations of
light sources using smartphone cameras of different brands.}
\label{fig:system_overview}
\end{figure}
```

### Figure 2: Color Change Grid
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/color_change_grid}
\caption{Images of $\mu$PADs showing visually observable color changes with varying
concentrations of [analyte] in [sample].}
\label{fig:colorchange}
\end{figure}
```

### Figure 3: Confusion Matrix
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.8\columnwidth]{figures/confusion_matrix}
\caption{Confusion matrix of [model name] for varying concentrations in the test dataset.}
\label{fig:confusion}
\end{figure}
```

### Figure 4: App Interface (Multi-Panel)
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/app_interface}
\caption{The steps for colorimetric [analyte] analysis in [AppName]. (a) Homepage,
(b) gallery selection, (c) crop interface, (d) ROI extraction, (e) analysis,
(f) results display with processing time.}
\label{fig:app}
\end{figure}
```

## Equations

### Performance Metrics (Always Include in Methods)

```latex
\begin{equation}
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\label{eq:accuracy}
\end{equation}

\begin{equation}
\text{Precision} = \frac{TP}{TP + FP}
\label{eq:precision}
\end{equation}

\begin{equation}
\text{Recall} = \frac{TP}{TP + FN}
\label{eq:recall}
\end{equation}

\begin{equation}
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\label{eq:f1}
\end{equation}

\begin{equation}
\text{LOD} = \frac{3\sigma}{\text{Slope}}
\label{eq:lod}
\end{equation}
```

Follow with: "where TP (True-Positive) and TN (True-Negative) describe the number of correctly identified positive and negative samples, while FP (False-Positive) and FN (False-Negative) define the incorrectly predicted samples."

## Quality Checklist

### Content
- [ ] Title follows group's pattern
- [ ] Abstract: 6-sentence structure, 150-200 words
- [ ] Keywords: 5-7 terms including smartphone, colorimetric, uPAD
- [ ] Introduction: 11-point flow
- [ ] Materials: chemicals with purity and suppliers
- [ ] uPAD fabrication: wax printing protocol (180 C, 120-180s, Whatman grade 1)
- [ ] Image acquisition: 7 lighting conditions, multi-phone, distance/angle
- [ ] AI method: architecture details, training parameters
- [ ] Results: performance tables, confusion matrix, SOTA comparison
- [ ] Conclusion: past tense, advantages list, applications

### Tables and Figures
- [ ] Table 1: Smartphone camera properties
- [ ] Table 2: ML/DL model comparison
- [ ] Table 3: Per-class metrics
- [ ] Table 4: State-of-the-art comparison
- [ ] All figures use `figures/filename.ext` relative paths
- [ ] All referenced images exist in `figures/` directory
- [ ] Figure source tracking comment at top of `.tex`

### LaTeX Quality
- [ ] Compiles without errors using IEEEtran class
- [ ] All cross-references work (\ref, \eqref, \cite)
- [ ] booktabs formatting for tables
- [ ] No absolute paths in \includegraphics
- [ ] All acronyms defined at first use
- [ ] Group self-citations included

### Overleaf Readiness
- [ ] .tex file in `assets/ieee_template/`
- [ ] IEEEtran.cls present (already included)
- [ ] All figures in `figures/` subdirectory
- [ ] Self-contained package ready to zip and upload
