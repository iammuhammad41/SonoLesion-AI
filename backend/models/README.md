# Model weights

Model checkpoints are intentionally excluded from Git.

Place your trained checkpoint here, for example:

```text
backend/models/sonolesion.pt
```

Then configure:

```env
SONOLESION_MODEL_PATH=backend/models/sonolesion.pt
```

The repository includes a generic MONAI 2-D U-Net adapter only to define the
deployment interface.

For an EALG-Net checkpoint or another custom architecture, replace
`build_model()` and, if necessary, `predict_with_checkpoint()` in:

```text
backend/services/inference.py
```

Do not commit patient data, restricted datasets, or weights that you do not
have permission to redistribute.
