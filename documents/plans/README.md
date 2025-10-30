# Implementation Plans Directory

This directory contains structured implementation plans created by the `plan-writer` agent and orchestrated by the `code-orchestrator` agent.

## Purpose

Plan files in this directory serve as:
- **Living documentation** of complex multi-phase implementations
- **Progress tracking** via checkboxes ([ ], [🔄], [✅], [⚠️])
- **Checkpoint management** for resuming work across sessions
- **Coordination artifacts** for multi-agent workflows

## File Naming Convention

- Format: `[TASK_NAME]_PLAN.md` (uppercase with underscores)
- Examples:
  - `AI_DETECTION_PLAN.md`
  - `REDIS_CACHING_PLAN.md`
  - `REFACTOR_PIPELINE_PLAN.md`

## When Plans Are Created

The `code-orchestrator` agent creates plan files for tasks that meet complexity criteria:
- **Multi-phase implementation** (3+ distinct phases)
- **Cross-file changes** (4+ files across directories)
- **Cross-language integration** (MATLAB + Python)
- **Complex refactoring** (core pipeline restructuring)
- **Long-running effort** (multiple sessions)
- **User-requested plan**

## Plan Structure

Each plan follows this template:
1. **Header**: Project overview, success criteria, hardware constraints
2. **Phases**: Logical groupings of related tasks
3. **Tasks**: Actionable items with code snippets, file paths, line numbers
4. **Test Cases**: Objective verification steps
5. **Progress Tracking**: Overall status, milestones, phase counts
6. **Notes & Decisions**: Design rationale, limitations, future work

## Version Control

Plans are living documents that must be kept synchronized with code:
- **Always commit plan updates with code changes**
- Update checkboxes as tasks complete
- Document design decisions in Notes section
- Never let plan drift from reality

Example commit:
```bash
git add augment_dataset.m documents/plans/AI_DETECTION_PLAN.md
git commit -m "Complete Phase 1.3: Export corner labels

- Implemented export_corner_labels() function
- Updated plan: Phase 1 (3/8 tasks)"
```

## Agents That Use This Directory

- **plan-writer**: Creates and updates plan files
- **matlab-coder**: Reads plans to understand implementation context
- **python-coder**: Reads plans to understand implementation context

## Reference Documentation

- Orchestration workflow: `CLAUDE.md` (Orchestration and Task Management section)
- Agent definitions: `.claude/agents/plan-writer.md`, `.claude/agents/matlab-coder.md`, `.claude/agents/python-coder.md`
- Project standards: `CLAUDE.md`
