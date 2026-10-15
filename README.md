# ST2M

Crystal structure generation with mixed discrete/continuous diffusion.

## 安装

在仓库根目录执行（外层目录可以任意命名）：

```bash
python -m pip install -e .
```

请先按照运行机器的 CUDA 环境安装合适的 PyTorch，再执行上面的命令。基础依赖由 `pyproject.toml` 声明，目前未锁定版本；训练复现应使用原来验证过的环境。MACE 验证功能另需 `python -m pip install -e ".[validation]"`。

## 集中入口

所有启动文件都放在 `scripts/`，对应实现放在 `st2m/`。在仓库根目录运行：

| 功能 | 命令 |
| --- | --- |
| 训练 | `python scripts/train.py` |
| 生成 | `python scripts/generate.py` |
| 生成反向扩散快照 | `python scripts/generate_steps.py` |
| 按新模式生成 | `python scripts/generate_new_pattern.py` |
| MP20 数据处理 | `python scripts/dataset_process_mp20.py` |
| 新模式模板采样 | `python scripts/sample_new_pattern.py` |
| 从 tensor 导出 CIF | `python scripts/extract_from_tensor.py` |
| DFT 后处理 | `python scripts/dft_post.py` |
| MACE 验证 | `python scripts/validate.py` |

这些 Python 入口也支持传入绝对路径，因此不依赖终端当前目录。训练和生成等带参数的入口可用 `--help` 查看参数；不带参数时使用 `st2m/utils/config.py` 中的默认值。DFT 后处理及 MACE 验证保留原有任务配置。

Windows 用户可以双击 `scripts/train.cmd`、`generate.cmd`、`generate_steps.cmd`、`generate_new_pattern.cmd`。启动器优先使用仓库内 `.venv/Scripts/python.exe`，否则使用 PATH 中的 `python`。使用 Conda 时建议在已激活环境的终端中运行 Python 入口；也可设置 `ST2M_PYTHON` 为解释器的完整路径。双击启动前，需要准备运行环境、数据和相应模型权重。启动器会执行实际任务，不会自动安装依赖。

安装后也可以在任意目录使用：

```bash
st2m-train
st2m-generate
python -m st2m.train
python -m st2m.generate
```

多 GPU 示例（在仓库根目录）：

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py
```

## 参数设置

默认参数集中在 `st2m/utils/config.py`，可直接在文件中修改，例如 `EPOCHS`、`BATCH_SIZE`、`LR`、`CHECKPOINT_PATH` 和 `SAMPLE_SPACE_GROUPS`。修改后直接运行对应脚本即可。命令行参数仅覆盖本次运行的默认值，例如 `python scripts/train.py --epochs 100`；无需在命令行重复填写其余参数。命令行参数注册代码单独放在 `st2m/cli_args.py`。

## 数据、权重与输出

默认位置保持为仓库根目录下的 `data/mp20/processed/`、`outputs/` 和 `temp/`。`mask_temp.pt` 位于 `st2m/utils/`，随 Python 包安装。训练和生成依赖的数据、检查点不由安装命令下载；请自行准备。现有默认检查点为 `outputs/checkpoints/best_no_CL.pt`，可通过 `--checkpoint-path` 覆盖。

可用环境变量覆盖数据和输出目录：

```powershell
$env:ST2M_DATA_ROOT = "D:/datasets/mp20/processed"
$env:ST2M_OUTPUT_DIR = "D:/runs/st2m"
python scripts/train.py
```

`ST2M_DATA_ROOT` 用于训练/生成读取处理后的数据；MP20 预处理仍从仓库 `data/mp20/` 读取原始数据并写入处理结果。普通安装到 site-packages 时建议显式设置数据和输出目录，避免向安装位置写入文件。

## 目录结构

```text
repository/
├── pyproject.toml
├── scripts/          # Python 入口及 Windows 双击启动器
├── st2m/             # 固定的 Python 包名
│   ├── train.py
│   ├── generate.py
│   ├── model/
│   ├── utils/
│   ├── preprocessing/
│   └── postproc/
├── data/
├── outputs/
└── manuscript/
```

原先 `from project...` 已统一改为 `from st2m...`。模型和算法不变。如果自行保存过包含完整 Python 模型对象（而非 state_dict）的旧 pickle 检查点，需另外处理其中记录的旧模块路径。
