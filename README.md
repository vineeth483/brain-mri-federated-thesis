# Brain MRI federated learning thesis

This private repository contains research code and selected results for four-class brain MRI image classification using centralized and federated learning.

## Browse the project

- `scripts/`: Python code for centralized training, dataset auditing, client partitioning, and Flower federated learning.
- `splits/`: dataset manifests and the IID and non-IID client assignments.
- `outputs/flower_iid_full/`: the IID run summary, validation history, starting weights, and selected model checkpoint. The `client_runs/` folder records each client and round.
- `outputs/trained_mobilenetv2/`: the centralized MobileNetV2 model and training results, including its earlier test evaluation.
- `requirements-mac-flower.txt`: direct Python dependencies for the Apple Silicon Flower environment. Some earlier evaluation scripts may need additional packages.

The IID federated checkpoint achieved **88.03% validation accuracy**. The federated test evaluation and non-IID training are still pending. The earlier centralized model used a different data split and is not a controlled comparison with this federated validation result.

The ZIP files are additional backups. The MRI images and unfinished progress report are excluded. The manifests expect the dataset in a `data/` folder; see the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and its terms before obtaining or redistributing it. These models are research artifacts, not clinically validated diagnostic tools.
