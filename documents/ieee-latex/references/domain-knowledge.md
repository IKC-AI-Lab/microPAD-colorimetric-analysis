# Domain Knowledge: microPAD Colorimetric Analysis Research Group

## Table of Contents

1. [Research Group Identity](#research-group-identity)
2. [Standard Terminology](#standard-terminology)
3. [Chemical and Enzymatic Language](#chemical-and-enzymatic-language)
4. [AI/ML Terminology](#aiml-terminology)
5. [Writing Style](#writing-style)
6. [Bibliography: Group Self-Citations](#bibliography-group-self-citations)
7. [Bibliography: Essential External References](#bibliography-essential-external-references)
8. [Experimental Design Patterns](#experimental-design-patterns)
9. [TODO Marker Guide](#todo-marker-guide)
10. [Fabrication Rules](#fabrication-rules)

---

## Research Group Identity

**Institution**: Izmir Katip Celebi University (IKCU), Izmir, Turkey
**Core team**: Kilic V., Sen M., Yuzer E., and collaborators
**Research focus**: AI-enhanced smartphone-based colorimetric sensing with microfluidic paper-based analytical devices (uPADs) for point-of-care diagnostics

**Hallmark**: Combining low-cost, accessible hardware with ML/DL for robust, phone-independent analysis.

**Signature features across publications**:
- Inter-phone repeatability (4+ smartphone brands)
- Illumination robustness (7 lighting conditions)
- Offline/embedded capability (TensorFlow Lite)
- User-friendly smartphone applications
- Point-of-care testing in resource-limited settings

## Standard Terminology

**Always use these exact terms**:

| Correct | Incorrect |
|---------|-----------|
| uPAD | microPAD, micro-PAD, upad |
| microfluidic paper-based analytical device | paper sensor, paper chip |
| point-of-care testing | POC (without definition) |
| smartphone-based / smartphone-embedded | phone-based, mobile-based |
| AI-based / ML-based / DL-based | AI powered (be specific to method type) |
| colorimetric quantification | color measurement |
| inter-phone repeatability | phone consistency |
| illumination variance | lighting changes |
| camera optics | camera quality |
| detection zone | sensing area, reaction spot |
| hydrophobic barriers | wax barriers |
| chromogenic agent/substrate | color indicator |
| feature extraction (ML) | data extraction |
| automatic feature learning (DL) | DL feature extraction |
| offline analysis | local analysis |
| resource-limited settings | low-resource environments |
| nonlaboratory environments | out-of-lab settings |

## Chemical and Enzymatic Language

Standard phrases for enzymatic detection:
- "GOx catalyzes the oxidation of beta-D-glucose to D-glucono-1,5-lactone"
- "H2O2 as a by-product"
- "HRP uses the by-product H2O2 to catalyze the conversion/oxidation of [chromogenic agent]"
- "forming a blueish/blue color change" (for TMB)
- "peroxidase-like activity" (if using chitosan)

Chemical notation: H2O2 (subscript in LaTeX: H$_2$O$_2$), beta-D-glucose ($\beta$-D-glucose), L-lactate

## AI/ML Terminology

**Classification vs. Regression** (always clearly distinguish):
- Classification: "categorizes data into predefined groups/classes," "assigns discrete categories"
- Regression: "determines the value of a dependent variable," "provides quantitative and continuous variables"

**Standard training setup**:
- Train/validation/test: 80:20, with 20% of training set as validation
- k-fold cross-validation: k=10 standard
- Hyperparameters: "epoch number, batch size, activation functions, optimizers, loss functions"
- Optimization: grid search
- Overfitting prevention: "early stopping," "regularization techniques"

**Performance metrics**:
- Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC
- Regression: MSE, MAE, R-squared, RMSE
- LOD: 3sigma/Slope or 3.3sigma/Slope
- Processing time: typically <1 second for embedded models

## Writing Style

**Tense**:
- Present tense: system descriptions ("The system demonstrates," "The model achieves")
- Past tense: what was done ("was developed," "were captured," "was embedded")
- Active voice preferred: "We developed" not "was developed by us"

**Standard opening phrases**:
- "Here, [contribution]..."
- "In this study, [method] was developed/proposed..."
- "Recently, [field] has attracted considerable attention..."
- "To address this issue, [solution]..."
- "The results demonstrated that..."

**Transition phrases**: However, Nevertheless, Additionally, Furthermore, Moreover, In this regard, As a result, Consequently, Therefore, Thus

**Result presentation**:
- "The system achieved/demonstrated/showed [metric] of [value]"
- "Maximum RMSE of X.XXX"
- "Classification accuracy of XX.X%"
- "LOD values of XXX uM for [analyte1] and XXX uM for [analyte2]"
- "Processing time of less than X s"

**Comparison language**: outperformed, superior performance, better than, improved over, comparable to, X% higher than, X-fold improvement

**Conclusion standard phrases**:
- "To the best of our knowledge, this is the first study..."
- "easy-to-use operation, rapid response, low-cost, high selectivity, consistent repeatability"
- "holds great promise for point-of-care testing"

## Bibliography: Group Self-Citations

Include these in every paper from this group:

```bibtex
@article{yuzer2022lactate,
  author  = {Y{\"u}zer, E. and Do{\u{g}}an, V. and K{\i}l{\i}{\c{c}}, V. and {\c{S}}en, M.},
  title   = {Smartphone embedded deep learning approach for highly accurate and automated
             colorimetric lactate analysis in sweat},
  journal = {Sens. Actuators B: Chem.},
  volume  = {371},
  pages   = {132489},
  year    = {2022}
}

@article{mercan2021glucose,
  author  = {Mercan, {\"O}.B. and K{\i}l{\i}{\c{c}}, V. and {\c{S}}en, M.},
  title   = {Machine learning-based colorimetric determination of glucose in artificial
             saliva with different reagents using a smartphone coupled $\mu${PAD}},
  journal = {Sens. Actuators B: Chem.},
  volume  = {329},
  pages   = {129037},
  year    = {2021}
}

@article{basturk2024regression,
  author  = {Ba{\c{s}}t{\"u}rk, M. and Y{\"u}zer, E. and {\c{S}}en, M. and K{\i}l{\i}{\c{c}}, V.},
  title   = {Smartphone-embedded artificial intelligence-based regression for colorimetric
             quantification of multiple analytes with a microfluidic paper-based analytical
             device in synthetic tears},
  journal = {Adv. Intell. Syst.},
  volume  = {6},
  pages   = {2400202},
  year    = {2024}
}
```

## Bibliography: Essential External References

```bibtex
@article{martinez2007patterned,
  author  = {Martinez, A.W. and Phillips, S.T. and Butte, M.J. and Whitesides, G.M.},
  title   = {Patterned paper as a platform for inexpensive, low-volume, portable bioassays},
  journal = {Angew. Chem. Int. Ed.},
  volume  = {46},
  number  = {8},
  pages   = {1318--1320},
  year    = {2007}
}

@article{carrilho2009wax,
  author  = {Carrilho, E. and Martinez, A.W. and Whitesides, G.M.},
  title   = {Understanding wax printing: a simple micropatterning process for paper-based
             microfluidics},
  journal = {Anal. Chem.},
  volume  = {81},
  number  = {16},
  pages   = {7091--7095},
  year    = {2009}
}

@article{kingma2014adam,
  author  = {Kingma, D.P. and Ba, J.},
  title   = {Adam: A method for stochastic optimization},
  journal = {arXiv preprint arXiv:1412.6980},
  year    = {2014}
}
```

## Experimental Design Patterns

**Multi-phone validation**: 4 smartphone brands (typically 2 iOS, 2 Android)

**Illumination conditions** (7 total):
- H: Halogen (2700K warm)
- F: Fluorescent (4000K neutral)
- S: Sunlight (6500K cold)
- HF, HS, FS: combinations
- HFS: all three combined

**Angle variations**: 30, 60, 90, 120, 150 degrees

**uPAD design for current project**:
- 7 test zones per strip
- 3 elliptical regions per zone
- Training: all 3 regions = same chemical/concentration
- Deployment: Region 1=urea, Region 2=creatinine, Region 3=lactate

**4-stage pipeline**:
1. `1_dataset` - raw smartphone images
2. `2_micropads` - cropped paper strips (via cut_micropads.m)
3. `3_concentration_regions` - polygonal test zones
4. `4_elliptical_regions` - elliptical patches for feature extraction

## TODO Marker Guide

Use these markers consistently for missing information:

| Marker | Purpose | Example |
|--------|---------|---------|
| `% TODO:` | General missing info | `% TODO: Determine number of phones` |
| `% TODO-USER:` | Needs user input | `% TODO-USER: Provide author names` |
| `% TODO-RESULTS:` | Needs experimental data | `% TODO-RESULTS: LOD values` |
| `% TODO-CITATION:` | Missing reference | `% TODO-CITATION: Find TMB reference` |
| `[TBD]` | In-text placeholder | `using [TBD] smartphone models` |
| `[TO BE MEASURED]` | Experimental value | `LOD of [TO BE MEASURED] uM` |
| `[XX]` | Numeric placeholder | `accuracy of [XX]\%` |

## Fabrication Rules

### IEEE Formatting (from IEEE Author Guidelines)

- Do NOT alter margins, column widths, line spaces, or text fonts
- Use vector formats (.eps, .pdf) for non-photo figures; avoid bitmapped (.jpeg, .png) for diagrams
- Number citations consecutively in square brackets [1]
- Do not use "Ref." or "reference" except at sentence start
- Unless 6+ authors, list all author names (no "et al." in references)
- Capitalize only first word in paper titles (except proper nouns)
- Label commands must come AFTER caption commands in figures/tables
- Title must be in boldface ALL CAPITALS
- 10pt font size (IEEEtran default)
