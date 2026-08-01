# Model setup

This archive is intentionally source-only. It does not contain model weights, downloaded Hugging Face files, datasets, or caches.

## Expected project-trained artifact

- Filename: `model_bundle_fp16.pt`
- Default relative path: `artifacts/models/vocal_activation/elgeish_baved_v1/model_bundle_fp16.pt`
- Expected size: 631,885,082 bytes (about 602.61 MiB)
- Manifest SHA-256: `d787347778c67f9b9f98322d3ef423c308f091e1eb678766f9f21e0517878380`
- Origin: project-trained deployment artifact derived from the BAVED candidate-training checkpoint recorded in `model_manifest.json`; it is not the original third-party backbone checkpoint.

The SHA value is provenance metadata, not an automatic runtime check. The current engine does not verify it before calling `torch.load`.

## Architecture

`src/inference/engine.py` constructs a Wav2Vec2-Large-XLSR-53 Arabic backbone (`elgeish/wav2vec2-large-xlsr-53-arabic`), followed by a one-layer bidirectional LSTM with hidden size 50 in each direction, dropout 0.2, mean/masked-mean pooling, and a 100-to-3 linear classifier. The three outputs are Low, Moderate, and High Vocal Activation.

The bundle stores 432 tensors as FP16. The current loader first constructs an ordinary FP32 model and then copies the state dictionary into it, so the smaller file does not establish an FP16 runtime-memory requirement.

## Resolution order

`resolve_model_path()` uses this order:

1. A path explicitly passed to `EmotionEngine`.
2. The `AUDIO_MODEL_PATH` environment variable.
3. The default relative path shown above.

Labels do not have an environment override. Keep `labels.json` at its supplied repository-relative path.

The application does not parse `.env` files. Set the variable in the shell or deployment platform, for example:

```powershell
$env:AUDIO_MODEL_PATH="path/to/authorized/model_bundle_fp16.pt"
streamlit run app.py
```

```bash
export AUDIO_MODEL_PATH="path/to/authorized/model_bundle_fp16.pt"
streamlit run app.py
```

If no artifact is present at any candidate path, `EmotionEngine` raises `FileNotFoundError`. The Streamlit application displays the loading failure and stops analysis; it does not fabricate predictions.

## Authorized reviewer procedure

1. Obtain the exact project-trained artifact through an authorized project channel. No public download location was verified, so none is supplied here.
2. Confirm that use and redistribution are authorized.
3. Verify its byte length and SHA-256 against the values above.
4. Place it at the default relative path, or set `AUDIO_MODEL_PATH` to the authorized local copy.
5. Ensure the backbone requirement below is satisfied.
6. Start the app from the project root.

Treat every `.pt` file as trusted code input. The current implementation uses `torch.load(..., weights_only=False)`, which can execute pickle content. Never load an artifact from an untrusted source.

## Additional Hugging Face requirement

The project bundle is not self-contained. During `load()`, the code calls `from_pretrained` for the backbone configuration, backbone model, and feature extractor. A clean machine therefore also needs either:

- permitted network access to resolve `elgeish/wav2vec2-large-xlsr-53-arabic`; or
- an authorized, already-populated compatible Hugging Face cache.

No downloaded cache is included in this submission, and offline startup was not established from this source-only archive.

## Licensing and redistribution

The repository's MIT license covers the project code but does not document redistribution rights for the Elgeish backbone, BAVED/YSED recordings, derived weights, or recorded voices. No complete third-party NOTICE/license inventory was found. Review the upstream model, dataset, and institutional permissions before providing the artifact or backbone cache to another person. This uncertainty is the reason the weight and all third-party caches are excluded.

