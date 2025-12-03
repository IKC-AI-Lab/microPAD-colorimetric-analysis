# review-matlab

Review MATLAB code for quality, style, and best practices using a complete review-fix-verify workflow

## Usage

```
/review-matlab [file_pattern]
```

## Description

Executes a comprehensive code quality workflow:
1. Reviews code using the matlab-code-reviewer agent
2. Fixes issues: simple fixes by Claude directly, complex fixes by matlab-coder agent
3. Re-reviews until all issues are resolved
4. Performs final verification with MATLAB Code Analyzer

The review covers:
- Code structure and readability
- Error handling and input validation
- Performance issues (array growth, pre-allocation)
- Documentation quality and accuracy
- MATLAB idioms and built-in function usage
- Project-specific conventions (naming, pipeline integration)
- Mask-aware feature computation patterns

## Workflow

When you invoke this command, follow this orchestration workflow:

### Phase 1: Initial Review
1. Delegate to `matlab-code-reviewer` agent to review the specified file(s)
2. Agent will identify issues and categorize by severity (Critical/High/Medium/Low)

### Phase 2: Fix Issues (if found)
3. If reviewer identifies issues:
   - **Simple fixes** (1-2 lines, obvious changes): Claude implements directly
   - **Complex fixes** (multi-function, algorithmic): Delegate to `matlab-coder` agent
4. Re-delegate to `matlab-code-reviewer` agent to verify fixes
5. Repeat steps 3-4 until review is clean

### Phase 3: Final Verification
6. Run MATLAB Code Analyzer on the file(s) using:
   ```bash
   matlab -batch "checkcode('file_path.m', '-id')"
   ```
7. Report any additional warnings from Code Analyzer
8. If Code Analyzer finds issues, return to Phase 2

### Phase 4: Completion
9. Confirm all reviews pass and Code Analyzer is clean
10. Provide summary of changes made and final quality assessment

## Examples

```bash
# Review a specific script
/review-matlab extract_features.m

# Review all helper scripts
/review-matlab helper_scripts/*.m

# Review entire matlab_scripts directory
/review-matlab matlab_scripts/
```

## Output

The command provides multi-phase output:

**Phase 1 - Initial Review:**
- Overall assessment of code quality
- Critical issues affecting functionality
- Code quality concerns and best practice violations
- Prioritized findings (Critical/High/Medium/Low)

**Phase 2 - Fixes (if needed):**
- Summary of issues being addressed
- Confirmation after each fix iteration
- Re-review results until clean

**Phase 3 - Final Verification:**
- MATLAB Code Analyzer results
- Any remaining warnings or suggestions
- Confirmation of clean analysis

**Phase 4 - Final Summary:**
- Complete list of changes made
- Final quality assessment
- Confirmation that code passes all checks

## Related Commands

- `/optimize-matlab` - Apply performance optimizations
- `/analyze-performance` - Deep performance analysis

## Notes

- This command follows the orchestration workflow defined in CLAUDE.md
- Multiple review-fix iterations may occur until code is clean
- MATLAB Code Analyzer provides final static analysis verification
- Simple fixes by Claude directly; complex fixes delegated to matlab-coder agent