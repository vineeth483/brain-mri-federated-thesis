# Brain MRI federated learning thesis

Private research backup. Download and extract all three ZIP files into the same folder to restore the directory structure.

- `code-and-results.zip`: source code, dependency file, data manifests and client assignments, selected metrics, validation history, and client-round records. It does not contain MRI images or the unfinished progress report.
- `federated-weights.zip`: selected IID model weights and the exact initialization for the future non-IID comparison.
- `centralized-model.zip`: the locally trained centralized MobileNetV2 model.

The full IID Flower run reached 88.03% on the validation split; the federated test evaluation and non-IID training are pending. See the README inside `code-and-results.zip` for further details.
