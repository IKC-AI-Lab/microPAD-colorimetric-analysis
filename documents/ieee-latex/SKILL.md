---
name: ieee-latex
description: >
  Write publication-quality IEEE LaTeX research papers using IEEEtran document class.
  Works for any IEEE conference or journal paper. Additionally specialized for microPAD
  colorimetric analysis from Izmir Katip Celebi University research group (Kilic, Sen,
  Yuzer et al.) with project-specific references for pipeline, YOLO detection, feature
  extraction, and literature review data. Use when the user wants to: (1) write or draft
  an IEEE conference or journal paper in LaTeX, (2) create technical documentation in IEEE
  format, (3) generate LaTeX content for colorimetric sensing, smartphone-based diagnostics,
  or microfluidic paper-based analytical device (uPAD) research, (4) format existing content
  into IEEE two-column layout, or (5) prepare Overleaf-ready LaTeX packages. Triggers on
  mentions of "IEEE paper", "LaTeX paper", "write paper", "conference paper", "research
  paper", "IEEEtran", "Overleaf", or requests involving scientific writing.
---

# IEEE LaTeX Paper Writer

Generate Overleaf-ready IEEE LaTeX documents. Works for any IEEE paper, with enhanced capabilities for the microPAD colorimetric analysis project.

## Quick Start

### For Any IEEE Paper
1. Read `references/paper-structure.md` for section templates and IEEE formatting rules
2. Read `references/domain-knowledge.md` for research group terminology
3. Copy `assets/ieee_template/` contents as the compilation base
4. Generate `.tex` file into `assets/ieee_template/` directory

### For microPAD Project Papers (Additional Steps)
5. Read `references/micropad-project.md` for pipeline details, features, and novelty framing
6. Read `references/literature-review.md` for SOTA comparison table and ready-to-use BibTeX

## Workflow

### Step 1: Clarify Requirements

Before writing, determine:
- **Document type**: `conference_paper` (6-8pp), `journal_article` (10-15pp), `technical_doc`, `thesis_chapter`
- **Analytes**: urea, creatinine, lactate, or all three
- **Authors**: names and emails (use TODO markers if unknown)
- **App name**: smartphone application name if applicable
- **Experimental status**: pre-experiment (template with TODOs) or post-experiment (with data)

### Step 2: Gather Project Facts

**For any paper** -- read from user's codebase (never fabricate):
- `README.md`, `CLAUDE.md` - project overview, experimental design
- Source scripts - parameter defaults, feature types

**For microPAD papers** -- additionally read:
- `references/micropad-project.md` - pipeline stages, feature categories, YOLO system, augmentation
- `references/literature-review.md` - 11 reviewed studies with BibTeX, 5 research gaps, SOTA table

### Step 3: Write Sections (Recommended Order)

1. **Materials and Methods** - most factual, extract from docs
2. **Introduction** - follow 11-point flow in `references/paper-structure.md`; for microPAD papers, frame novelty around 5 research gaps from `references/literature-review.md`
3. **Results and Discussion** - include standard tables/figures (or TODO markers); use SOTA comparison data from literature review
4. **Abstract** - write last, condense methods+results into ~150-200 words
5. **Conclusion** - restate contribution, list advantages
6. **Bibliography** - include group self-citations from `references/domain-knowledge.md`; add literature review BibTeX from `references/literature-review.md`

### Step 4: Figure Management

Copy images from pipeline outputs to `assets/ieee_template/figures/`:
- `demo_images/` - representative pipeline stage examples
- `3_concentration_regions/{phone}/con_*/` - concentration series
- All `\includegraphics` must use `figures/filename.ext` (relative paths)

### Step 5: Output

- **Filename**: `microPAD_colorimetric_analysis_[topic].tex`
- **Location**: `assets/ieee_template/` (Overleaf-ready)
- **Document class**: `\documentclass[conference]{IEEEtran}`
- Compile: `pdflatex` twice for cross-references

## Critical Rules

### Never Fabricate
- No invented results, LOD values, accuracy percentages, or metrics
- No made-up author info, smartphone specs, or dataset sizes
- Use `% TODO-RESULTS:`, `% TODO-USER:`, `% TODO-CITATION:`, `[TBD]`, `[XX]` for unknowns

### Always Do
- Use group's exact terminology (see `references/domain-knowledge.md`)
- Define all acronyms at first use
- Include standard tables and figures (with TODO markers if data missing)
- Include group self-citations in bibliography
- Track figure sources with comments at top of `.tex` file

## Reference Files

| File | When to Read | Content |
|------|-------------|---------|
| `references/paper-structure.md` | Always | Section templates, table/figure patterns, equations, quality checklist |
| `references/domain-knowledge.md` | Always | Terminology, bibliography entries, writing style, research patterns |
| `references/micropad-project.md` | microPAD papers | Pipeline architecture, feature system, YOLO detection, augmentation, novelty framing, demo images, suggested titles |
| `references/literature-review.md` | microPAD papers | SOTA comparison table (11 studies), 5 research gaps, trend analysis, complete BibTeX for all reviewed papers |

## Assets

| Path | Purpose |
|------|---------|
| `assets/ieee_template/IEEEtran.cls` | IEEE document class (do not modify) |
| `assets/ieee_template/IEEEtran.bst` | IEEE bibliography style |
| `assets/ieee_template/ieee_template.tex` | Reference template from group |

## microPAD Project Capabilities

When writing papers for this specific project, the skill provides:

- **Novelty positioning**: 5 identified research gaps mapped to project contributions (see `references/literature-review.md` and `references/micropad-project.md`)
- **Ready-to-use SOTA table**: Verified data from 11 systematically reviewed papers (2017-2024)
- **Complete BibTeX**: All literature review references + group self-citations (15 papers, 2016-2024)
- **Pipeline documentation**: 4-stage architecture with exact coordinate formats, MATLAB commands
- **Feature engineering details**: 150+ features across 13 categories with preset descriptions
- **YOLO detection system**: Architecture, training config, MATLAB integration commands
- **Augmentation pipeline**: Two-stage MATLAB+YOLO architecture with performance specs
- **Demo images inventory**: 9 pipeline stage examples ready to copy to `figures/`
- **Suggested paper titles**: 4 title templates following group's naming pattern
