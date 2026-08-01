# Elgeish Baved Vocal Activation Model — v1.0.0

## Purpose

Deployment artifact for Arabic Speech Emotion Recognition (SER) using the Elgeish standalone architecture trained on the BAVED dataset.

## Architecture

```
Elgeish Wav2Vec2-BiLSTM Standalone
├── Backbone: elgeish/wav2vec2-large-xlsr-53-arabic
├── BiLSTM: input=1024, hidden=50, layers=1, bidirectional
├── Pooling: masked mean pooling
├── Dropout: 0.2
└── Classifier: Linear(100, 3)
```

## Official Labels

| Index | Label |
|-------|-------|
| 0 | Low Vocal Activation |
| 1 | Moderate Vocal Activation |
| 2 | High Vocal Activation |

## Source Checkpoint

- **Path:** `experiments/emotion_type/candidate_screening/elgeish_baved/01_training_implementation/outputs/checkpoint-1695/model.safetensors`
- **Format:** SafeTensors, FP32
- **Size:** 1,263,531,004 bytes (1.18 GB)
- **Tensor count:** 432

## Deployment Artifact

- **Path:** `artifacts/models/vocal_activation/elgeish_baved_v1/model_bundle_fp16.pt`
- **Format:** PyTorch bundle, FP16
- **Size:** 631,885,082 bytes (602.61 MB)
- **Tensor count:** 432
- **SHA-256:** `d787347778c67f9b9f98322d3ef423c308f091e1eb678766f9f21e0517878380`

## Validation Status

- ✅ Strict state-dict loading: PASSED
- ✅ Key identity: PASSED (zero missing, zero unexpected)
- ✅ Shape identity: PASSED (zero mismatches)
- ✅ FP16 conversion accuracy: PASSED (all 432 tensors exact match)
- ✅ Config field presence: PASSED
- ✅ Config value correctness: PASSED
- ✅ Official labels: PASSED

## Evaluation Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.5986 |
| Test Macro F1 | 0.6007 |
| Evaluation Protocol | Speaker-disjoint test |

## Known Limitation

High Vocal Activation (class 2) is frequently confused with Moderate Vocal Activation (class 1). This is an expected model behavior given the acoustic similarity between these classes in the BAVED dataset.

## Current Integration Status

**Integrated and Validated.**

This artifact has been validated for structural integrity, FP16 conversion accuracy, logits parity, real inference, end-to-end pipeline correctness, and Streamlit startup.

### Validation Results (2026-07-31)

| Check | Status |
|-------|--------|
| Strict state-dict loading | ✅ PASSED |
| Masked pooling parity | ✅ PASSED |
| FP32/FP16 logits parity | ✅ PASSED |
| Real inference smoke test | ✅ PASSED |
| End-to-end pipeline | ✅ PASSED |
| Streamlit startup | ✅ PASSED |
| Focused test suite | ✅ 154/154 passed |

**Tested runtime target:** Streamlit
