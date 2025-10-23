# write-ieee-latex

Generate professional IEEE-formatted LaTeX documentation for the microPAD colorimetric analysis project

## Usage

```
/write-ieee-latex [document_type]
```

## Description

Launches the IEEE LaTeX Documentation Writer agent to create comprehensive technical documentation covering:
- Research background (microPAD technology, colorimetric analysis, biomarker detection)
- Technical architecture (5-stage pipeline, data flow, coordinate management)
- Experimental design (7 test zones, 3 regions, multi-phone dataset, lighting conditions)
- Implementation details (memory optimization, feature extraction, augmentation)

The agent generates a complete, compilable LaTeX document ready for conversion to PDF.

## Parameters

**document_type** (optional): Document type
- `conference_paper` - IEEE conference paper format (default, two-column, 6-8 pages)
- `journal_article` - IEEE journal article format (two-column, 10-15 pages)
- `technical_doc` - Technical documentation with article class
- `thesis_chapter` - Thesis chapter format with book/report class

## Examples

```bash
# Generate IEEE conference paper (default)
/write-ieee-latex

# Generate IEEE journal article
/write-ieee-latex journal_article

# Generate technical documentation
/write-ieee-latex technical_doc

# Generate thesis chapter
/write-ieee-latex thesis_chapter

# With author information
/write-ieee-latex conference_paper authors='Elif Yüzer <elif.yuzer@ikcu.edu.tr>; Volkan Kılıç <volkan.kilic@ikcu.edu.tr>'
```

## Output

**File**: `microPAD_[analyte]_IEEE_paper.tex` (in `documents/ieee_template/` directory)

**Content includes** (IEEE Conference Format):
1. Title and author block (IEEEtran format)
2. Abstract (6-sentence pattern from research group's publications)
3. Keywords (5-7 terms)
4. Introduction (11-point structure following group's pattern)
5. Experimental Section / Materials and Methods
   - Materials and chemicals
   - μPAD design and fabrication (wax printing protocol)
   - Image acquisition (multi-phone, multi-illumination setup)
   - AI-based classification/regression methodology
   - Smartphone application (if applicable)
6. Results and Discussion
   - Sensor performance and LOD
   - Model comparison tables
   - Smartphone app demonstration
   - Real sample testing (if applicable)
   - Selectivity results
   - State-of-the-art comparison
7. Conclusion
8. Acknowledgements (Turkey funding format)
9. Author contributions
10. References (with group's self-citations)

**Features**:
- IEEE IEEEtran document class (conference or journal format)
- Research group's exact terminology (μPAD, inter-phone repeatability, etc.)
- Standard tables (smartphone properties, model comparison, state-of-the-art)
- Standard figures (system schematic, color change grid, confusion matrix, app screenshots)
- Performance metrics equations (Accuracy, Precision, Recall, F1-score, LOD)
- TODO markers for missing experimental data (never fabricates results)
- Cross-references and citations
- Author parameter support for automatic author block generation

**Length**: 6-8 pages for conference paper, 10-15 pages for journal (when compiled)

## Compilation

Compile the generated LaTeX file to PDF:

```bash
# Navigate to documents/ieee_template/ directory
cd documents/ieee_template/

# Standard compilation (run twice for cross-references)
pdflatex microPAD_[analyte]_IEEE_paper.tex
pdflatex microPAD_[analyte]_IEEE_paper.tex
```

**Requirements**:
- Standard LaTeX distribution (TeX Live, MiKTeX, MacTeX)
- IEEEtran.cls file (already in documents/ieee_template/)

**Packages used**: cite, amsmath, amssymb, amsfonts, algorithmic, graphicx, textcomp, xcolor, booktabs, hyperref

## Use Cases

- **Conference submissions**: IEEE conference papers ready for submission
- **Journal manuscripts**: IEEE Transactions or Sensors and Actuators B format
- **Research documentation**: Following Izmir Katip Celebi University research group's style
- **Grant proposals**: Technical background and methodology sections
- **Thesis chapters**: Adapted from group's publication pattern

## Customization

After generation, you can customize:
- **TODO markers**: Replace all TODO comments with actual experimental data
- **Author block**: Provide authors parameter to auto-generate (or edit manually)
- **Results**: Fill in performance metrics (accuracy, LOD, RMSE) from experiments
- **Figures**: Replace placeholders with actual images from experiments
- **Bibliography**: Complete TODO-CITATION markers with full references
- **Analyte focus**: Modify sections to emphasize specific analytes (urea/creatinine/lactate)

## Tips

**Before running**:
- Decide on document type (conference_paper vs. journal_article vs. technical_doc)
- Determine which analytes to focus on (urea, creatinine, lactate, or all)
- Gather author information if available (names and emails)
- Check if experimental results are available (or document will use TODO markers)

**Providing Parameters**:
```bash
# With authors
/write-ieee-latex conference_paper authors='Name1 <email1@ikcu.edu.tr>; Name2 <email2@ikcu.edu.tr>'

# With analyte specification
/write-ieee-latex conference_paper analytes='lactate'

# With app name
/write-ieee-latex conference_paper app_name='ChemiCheck'

# Multiple parameters
/write-ieee-latex journal_article authors='...' analytes='all' app_name='ChemiCheck'
```

**After generation**:
- Search for all `% TODO` comments and address them
- Replace `[TBD]` and `[TO BE MEASURED]` with actual values
- Add actual image files and update `\includegraphics` paths
- Complete bibliography entries marked with `% TODO-CITATION`
- Compile with `pdflatex` and check for errors

**Anti-Fabrication Guarantee**:
- The agent **NEVER** invents experimental results or performance metrics
- All missing data marked with clear TODO comments
- Questions asked when critical information is unclear
- Only factual information from documentation is included

## Related Commands

- `/review-matlab` - Review MATLAB code quality before documenting
- Standard git commands - Commit documentation to version control

## Specialized Features

**Research Group Style Matching**:
- Based on 15+ previous publications (2016-2024) from Izmir Katip Celebi University
- Exact terminology: μPAD, inter-phone repeatability, illumination variance
- Standard experimental design: 4 phones × 7 lighting conditions × 5 angles
- Established paper structure: 6-sentence abstract, 11-point introduction
- Standard figures: system schematic, color change grid, confusion matrix, app screenshots
- Performance metrics: Accuracy, Precision, Recall, F1-score, MSE, MAE, R², RMSE, LOD

**Parameter Support**:
- `authors`: Semicolon-separated list with format 'Name <email>; Name2 <email2>'
- `document_type`: conference_paper, journal_article, technical_doc, thesis_chapter
- `analytes`: urea, creatinine, lactate, or 'all'
- `app_name`: Smartphone application name (e.g., ChemiCheck, DeepLactate)

**Quality Assurance**:
- 91-point comprehensive quality checklist
- Verification against group's lexicon and terminology
- Ensures IEEE formatting compliance
- Cross-references to template files (IEEEtran.cls, conference_101719.tex)
