# ST2M

ST2M generates crystal structures using mixed discrete and continuous diffusion.

## Installation

Clone or download the repository, open a terminal in its root directory, and create a dedicated Python environment. The name of the downloaded repository directory does not matter.

Install a PyTorch build compatible with your CUDA environment first. Then install ST2M and its core dependencies:

```bash
python -m pip install -e .
```

The editable installation registers the `st2m` package and its command-line entry points in the active Python environment. Changes made to the source code or `st2m/utils/config.py` are used immediately, so the package normally does not need to be reinstalled after source edits.

To install the optional dependencies used for structure validation:

```bash
python -m pip install -e ".[validation]"
```

Dependency versions are currently not pinned in `pyproject.toml`. Use a tested environment with recorded package versions when exact reproduction is required.

## Running ST2M

Run the following commands from the repository root:

| Task | Command |
| --- | --- |
| Train the model | `python scripts/train.py` |
| Generate crystal structures | `python scripts/generate.py` |
| Save reverse-diffusion snapshots | `python scripts/generate_steps.py` |
| Generate structures from new patterns | `python scripts/generate_new_pattern.py` |
| Process the MP20 dataset | `python scripts/dataset_process_mp20.py` |
| Sample new pattern templates | `python scripts/sample_new_pattern.py` |
| Export CIF files from saved tensors | `python scripts/extract_from_tensor.py` |
| Run DFT post-processing | `python scripts/dft_post.py` |
| Run structure validation | `python scripts/validate.py` |

Use `--help` to inspect the available command-line options for an entry point:

```bash
python scripts/train.py --help
python scripts/generate.py --help
```

After installation, the entry points declared in `pyproject.toml` can also be called from any directory:

```bash
st2m-train
st2m-generate
st2m-generate-steps
st2m-generate-new-pattern
st2m-process-mp20
st2m-sample-new-pattern
st2m-extract
```

The equivalent module commands are also available:

```bash
python -m st2m.train
python -m st2m.generate
```

For multi-GPU training, run the launcher from the repository root. For example:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py
```

### Windows launchers

Windows users can run `train.cmd`, `generate.cmd`, `generate_steps.cmd`, or `generate_new_pattern.cmd` from the `scripts` directory. Each launcher first looks for `.venv/Scripts/python.exe` in the repository and otherwise uses the `python` command available on `PATH`.

When using Conda, running the Python entry point from an activated environment is recommended. Alternatively, set `ST2M_PYTHON` to the full path of the desired Python executable before launching a `.cmd` file.

The launchers start the requested task; they do not install dependencies, datasets, or model weights.

## Configuration

Default settings are defined in `st2m/utils/config.py`. They can be edited directly before running a task. Common settings include:

```python
EPOCHS = 1000
BATCH_SIZE = 32
LR = 2e-4
CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pt"
SAMPLE_SPACE_GROUPS = (...)
```

When no command-line option is supplied, the value in `config.py` is used. A command-line option overrides that value for one run without modifying the file:

```bash
python scripts/train.py --epochs 100 --batch-size 16
```

Command-line argument definitions are kept in `st2m/cli_args.py`, while editable experiment defaults remain in `st2m/utils/config.py`.

## Data, checkpoints, and outputs

The default locations are relative to the repository root:

```text
data/mp20/processed/       processed training, validation, and test data
outputs/checkpoints/       model checkpoints
outputs/samples/           generated samples
temp/                      reverse-diffusion snapshots
```

The default generation checkpoint is:

```text
outputs/checkpoints/best.pt
```

Datasets and checkpoints are not downloaded by the installation command. They must be prepared or obtained separately before training or generation.

The data and output roots can be changed with environment variables. In PowerShell:

```powershell
$env:ST2M_DATA_ROOT = "D:/datasets/mp20/processed"
$env:ST2M_OUTPUT_DIR = "D:/runs/st2m"
python scripts/train.py
```

`ST2M_DATA_ROOT` changes the processed dataset location used by training and generation. `ST2M_OUTPUT_DIR` changes the location used for checkpoints, logs, and generated samples.

The MP20 preprocessing entry point currently reads raw data from `data/mp20/` in the repository and writes the processed dataset there. The required `mask_temp.pt` resource is stored in `st2m/utils/` and is included when the package is built or installed.

## Repository structure

```text
repository/
├── pyproject.toml
├── README.md
├── scripts/                 task entry points and Windows launchers
├── st2m/                    importable Python package
│   ├── train.py
│   ├── generate.py
│   ├── cli_args.py
│   ├── model/
│   ├── utils/
│   ├── preprocessing/
│   └── postproc/
├── data/
├── outputs/
└── manuscript/
```

The repository directory may have any name because internal imports use the fixed package name `st2m`.

Older checkpoints that contain only model and optimizer state dictionaries remain independent of the old module path. A legacy pickle containing complete Python model objects may still reference the former `project` package path and require conversion before it can be loaded.
