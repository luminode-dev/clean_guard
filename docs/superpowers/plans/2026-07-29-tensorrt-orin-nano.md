# TensorRT Orin Nano Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing two-model export flow reliably build FP16 TensorRT engines on Jetson Orin Nano.

**Architecture:** Keep Ultralytics `.pt` export as the primary path and add a deterministic `trtexec` ONNX fallback. The script will validate Jetson prerequisites, use batch-1/640 input and bounded workspace, and verify each produced engine.

**Tech Stack:** Bash, Python 3, Ultralytics YOLO, TensorRT `trtexec`, JetPack 6.x.

## Global Constraints

- Build engines on the target Jetson; engines are GPU/ TensorRT-version specific.
- Use FP16, image size 640, batch size 1, and a 2048 MiB workspace cap.
- Do not enable INT8 without a calibration dataset and accuracy validation.

### Task 1: Add testable conversion-script contract

**Files:**
- Create: `tests/test_convert_tensorrt.sh`
- Modify: `convert_tensorrt.sh`

- [ ] Write tests asserting shell syntax, both model names, FP16, batch-1/640 settings, workspace cap, and engine verification.
- [ ] Run the tests and confirm they fail against the current script because the required contract is absent.
- [ ] Implement prerequisite checks, configurable `TRT_WORKSPACE_MB`, robust output handling, and `trtexec` fallback.
- [ ] Run tests and shell syntax validation.

### Task 2: Document Orin Nano deployment

**Files:**
- Modify: `README.md`

- [ ] Document prerequisites, command usage, environment overrides, fallback behavior, and the fact that engine creation must happen on the Orin Nano.
- [ ] Verify all documented filenames and options match the script.

### Task 3: Final verification

**Files:**
- No production files.

- [ ] Run shell tests and inspect the final diff.
- [ ] Report that actual engine generation requires the Jetson TensorRT runtime if unavailable in this workspace.
