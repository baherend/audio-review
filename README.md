# Audio Doctor — Stereo Call Audio Evidence Review

## 1. Project overview

Audio Doctor is a Streamlit prototype for Team Leaders reviewing recorded Arabic customer-service calls. It analyzes a two-channel WAV file and presents time-aligned audio evidence: channel activity, per-speaker vocal-activation distributions, prototype Hold candidates, Dead Air/overlap alerts, and a consolidated Final Call Review.

## 2. Evidence-only scope

The application works only from acoustic and timing evidence. It does not transcribe speech, run ASR or NLP, interpret words, identify people, infer intent, or verify which real person is on a channel.

## 3. Main features

- Strict stereo loading and channel-quality warnings.
- Resampling to the model's 16 kHz rate.
- Shared 2-second windows with a 1-second hop.
- RMS-based per-channel activity gating.
- Three-class Vocal Activation inference for active windows.
- Agent and Customer summaries plus an exact one-second activity timeline.
- Heuristic Hold candidates and a configurable prototype Hold policy.
- Dead Air, overlap, Hold, and duration alert machinery.
- Evidence aggregation into a Final Call Review.
- Downloadable technical JSON and a Team Leader PDF report.

## 4. Architecture summary

```text
app.py
  -> stereo_loader + channel_validation
  -> temporal_analysis + speaker_processor + EmotionEngine
  -> speaker_summary
  -> hold_detection + hold_policy
  -> operational_alerts
  -> call_review
  -> Streamlit display + JSON/PDF export
```

The Streamlit orchestrator is `run_pipeline()` in `app.py`. Dataclasses in `src/demo/schemas.py` define the pipeline contracts; `src/inference/engine.py` owns model resolution, loading, and prediction.

## 5. Folder structure

```text
.
├── app.py                         Streamlit entry point and orchestration
├── src/
│   ├── inference/                 Model architecture, loader, inference schemas
│   └── demo/                      Audio, activity, Hold, alert, review, PDF logic
├── tests/                         Canonical reviewer and regression tests
├── artifacts/models/vocal_activation/elgeish_baved_v1/
│   ├── labels.json                Runtime label mapping
│   ├── model_manifest.json        Model provenance and historical metrics
│   ├── README.md                  Artifact metadata
│   └── SHA256SUMS                 Expected weight checksum; weight not included
├── MODEL_SETUP.md                 Authorized local model procedure
├── requirements.txt               Runtime and test dependencies
├── .env.example                   Non-secret environment-variable example
└── LICENSE                        Project code license
```

No dataset, audio sample, model weight, environment, cache, experiment output, or downloaded library is bundled. A demo recording was not included because legal provenance and privacy safety were not established.

## 6. Audio requirements

The UI accepts `.wav` files and the loader requires exactly two channels. It accepts the source sample rate reported by the decoder and resamples each channel to 16 kHz. Mono and more-than-stereo inputs are rejected. Non-finite audio and a call with both channels fully silent are rejected; identical channels, one silent channel, clipping, low energy, and very short calls generate warnings in relevant cases.

There is no explicit application-level duration or upload-size limit. Use only recordings that you are authorized to process.

## 7. Stereo mapping

- Left Channel = Agent
- Right Channel = Customer

This is a fixed file-format convention, not speaker diarization. The program cannot prove the channels were recorded or labeled correctly.

## 8. Official labels

- Low Vocal Activation
- Moderate Vocal Activation
- High Vocal Activation

These are acoustic activation classes, not emotions, sentiment, satisfaction, conduct, or performance grades.

## 9. Installation

Python 3.10 or 3.11 is the conservative reviewer target reflected by the inspected environments. From the project root:

```bash
python -m venv .venv
```

Activate the environment using the command appropriate for your platform, then install:

```bash
python -m pip install -r requirements.txt
```

Dependencies were not installed while producing this submission. The requirements use compatible ranges rather than a fully locked environment, so exact reproducibility is not guaranteed.

## 10. Environment setup

`AUDIO_MODEL_PATH` is the only application-specific runtime environment variable. It is optional when the model is placed at the default path. `.env.example` is documentation only: the code does not automatically parse `.env` files.

For privacy and security, keep local `.env` files untracked. Do not place credentials in this application; none are required by the source.

## 11. Model setup

The model weight is deliberately excluded. Follow `MODEL_SETUP.md` for the exact filename, expected size, checksum, path-resolution order, trust warning, and the additional Hugging Face backbone/cache requirement. Do not substitute an unrelated `.pt` file.

## 12. Run the Streamlit application

After dependencies and the authorized model/backbone are available, run from the project root:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, enter an optional Call ID, upload an authorized stereo WAV file, and start analysis. The current UI hardcodes call type to `unknown`, so call-duration policy comparisons are not active in the normal app flow.

## 13. Run tests

For the included model-independent reviewer subset:

```bash
python -B -m pytest -q -p no:cacheprovider tests/test_hold_policy.py tests/test_operational_alerts.py tests/test_call_review.py tests/test_pdf_report.py tests/parity/test_masked_pooling_parity.py tests/test_app_integration.py::TestAppContent tests/test_app_integration.py::TestAppUIPolish tests/test_app_integration.py::TestSidebarPolish tests/test_app_integration.py::TestMainPageVisualPolish tests/test_app_integration.py::TestExportSummary
```

Before running, set `PYTHONDONTWRITEBYTECODE=1` and, if strict offline validation is intended, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The full suite is:

```bash
python -B -m pytest -q -p no:cacheprovider
```

Model/inference tests require the intentionally excluded project artifact and a resolvable backbone. A source-only failure for those tests is expected and must not be reported as a product pass.

## 14. Expected outputs

The page shows input/channel validation, Agent and Customer activation summaries, a shared activity timeline, inactivity evidence, Hold review, operational alerts, and a Final Call Review with reasons and limitations. Results depend on the supplied audio and model.

## 15. JSON export

After a successful analysis, use the JSON download button. The JSON is the technical serialized `FinalCallReview` evidence object. It can contain call identifiers and operational timing evidence, so handle it as potentially sensitive data.

## 16. PDF export

Use the PDF download button to generate the evidence-only Team Leader report built by `src/demo/pdf_report.py`. PDF generation requires ReportLab. The PDF is a presentation of the aggregated review; it is not an independent second analysis.

## 17. Known limitations

- A clean machine is not fully offline-capable: the engine still calls Hugging Face `from_pretrained` even when the project bundle is supplied.
- The active speaker detector is an RMS heuristic, not a trained VAD or diarizer.
- Hold music and recorded-message evidence are prototype spectral heuristics, not validated detectors; Hold thresholds are not an official policy.
- The Hold autocorrelation implementation is quadratic on up to 80,000 samples and can be slow.
- High Vocal Activation is frequently confused with Moderate; the supplied manifest reports 0.5986 accuracy and 0.6007 macro-F1 on its speaker-disjoint evaluation.
- Per-window inference exceptions can be silently omitted from summaries.
- Invalid decoded uploads can reach a Streamlit `KeyError` in the current error-display path.
- Streamlit session results are not keyed to the selected upload and can remain stale until a new run completes.
- Temporary audio is written under the project directory and normally deleted, but a hard process failure could leave it behind.
- Review status/manual-review fields and exported Hold counting mode have known consistency defects documented in the external audit.
- PDF values below one percent and markup-like Call IDs have known formatting/robustness issues.
- No explicit resource limit, authentication, encryption, retention policy, or telecom integration exists.

## 18. Privacy notice

Process recordings only with an appropriate legal basis and institutional authorization. Uploaded audio is decoded into memory and temporarily written to disk; result objects can retain original and resampled waveforms in the Streamlit session. JSON/PDF exports may contain identifiers and call evidence. The application provides no access control, multi-user isolation guarantee, automatic redaction, retention enforcement, or secure deletion. Close the session and inspect local temporary state according to your organization's policy.

## 19. Troubleshooting

- **Model not found:** place the authorized artifact at the default relative path or export `AUDIO_MODEL_PATH`; see `MODEL_SETUP.md`.
- **Backbone cannot be resolved:** provide permitted network access or an authorized compatible Hugging Face cache. The project weight alone is insufficient.
- **Mono/channel error:** export the recording as a true two-channel WAV with the required left/right convention.
- **Analysis stops after model setup:** restart Streamlit or clear its resource cache because a failed model load may have been cached.
- **PDF import error:** confirm ReportLab is installed from `requirements.txt`.
- **CPU run is slow or memory-heavy:** the Wav2Vec2-Large backbone is substantial and long calls are analyzed sequentially; no minimum hardware benchmark is certified.

## 20. Disclaimer

This system provides review evidence only. It does not produce official QA scores, Pass/Fail decisions, employee accusations, sentiment judgments, or final management decisions. Its outputs require review by an authorized human who considers the original call, applicable policy, and context.

## License and submission note

Project code is supplied under the repository's MIT license. The license file does not name a rights holder, and it does not establish permission to redistribute datasets, voices, third-party models, or derived weights. The application source in this package is copied unchanged from the audited canonical implementation; the README, model instructions, dependency list, environment example, and ignore rules are submission-specific replacements.

