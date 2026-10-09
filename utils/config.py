from __future__ import annotations

from pathlib import Path

NUM_ELEMENT_CLASSES = 100
NUM_DISCRETE_CLASSES = NUM_ELEMENT_CLASSES + 1
VACANCY_CLASS = 0
DEFAULT_MAX_INF_OCC = 30
MAX_WYCKOFF_SITES = 27

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "mp20" / "processed"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SAMPLE_DIR = OUTPUT_DIR / "samples"
SPACEGROUP_TEMPLATE_PATH = PROJECT_ROOT / "utils" / "mask_temp.pt"
TRAIN_LOG_FILE = OUTPUT_DIR / "train.log"
TEST_LOG_FILE = OUTPUT_DIR / "test.log"

LATTICE_DIM = 6

SEED = 42
DEVICE = "auto"
NUM_WORKERS = 0
PIN_MEMORY = True

EPOCHS = 1000
BATCH_SIZE = 32
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LOG_EVERY = 50
VAL_EVERY = 15
SAVE_EVERY = 10
MAX_TRAIN_SAMPLES = 0
MAX_VAL_SAMPLES = 0
MAX_TEST_SAMPLES = 0

TIMESTEPS = 1000
DISCRETE_SCHEDULE = "cosine"
CONTINUOUS_SCHEDULE = "linear"
BETA_START = 1e-4
BETA_END = 2e-2
DISCRETE_LOSS_WEIGHT = 1.0
CONTINUOUS_LOSS_WEIGHT = 1.0
LATTICE_LOSS_WEIGHT = 1.0
ORBIT_COUNT_LOSS_WEIGHT = 0.01
MULTIPLICITY_COUNT_LOSS_WEIGHT = 0.01
USE_COUNT_LOSSES = False
COUNT_LOSS_ORBIT_MARGIN = 0.5
COUNT_LOSS_ATOM_MARGIN = 2.0
COUNT_LOSS_SMOOTH_TAU = 0.5
LAMBDA_CONTRASTIVE_CON = 0.1
LAMBDA_CONTRASTIVE_DIS = 0.1
CONTRASTIVE_MARGIN = 1.0
CONTRASTIVE_USE_X0 = False

MODEL_TYPE = "wyckoff_transformer"
TRANSFORMER_D_MODEL = 256
TRANSFORMER_NHEAD = 8
TRANSFORMER_LAYERS = 6
TRANSFORMER_FF_DIM = 1024
TRANSFORMER_DROPOUT = 0.0
TRANSFORMER_DENOISER_MODE = "separate"

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_no_CL.pt"
RESUME = ""
RESUME_BEST = False
EVAL_SPLIT = "test"
NUM_SAMPLE_BATCHES = 50
SAMPLE_BATCH_SIZE = 200
SAMPLE_SPACE_GROUP = 0
SAMPLE_SPACE_GROUPS: tuple[int, ...] = (225,12,139,62,194,166,63,221,2,123,14,164,216,15,129,1,189,71,38,8,148)
SAVE_SAMPLES = True
SAVE_CIFS = True
NUM_SAVE_CIFS = 9760


def add_common_args(parser):
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=PIN_MEMORY)
    parser.add_argument("--train-dir", type=str, default=str(TRAIN_DIR))
    parser.add_argument("--val-dir", type=str, default=str(VAL_DIR))
    parser.add_argument("--test-dir", type=str, default=str(TEST_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--sample-dir", type=str, default=str(SAMPLE_DIR))
    parser.add_argument("--lattice-dim", type=int, default=LATTICE_DIM)
    parser.add_argument("--num-discrete-classes", type=int, default=NUM_DISCRETE_CLASSES)
    parser.add_argument("--vacancy-class", type=int, default=VACANCY_CLASS)
    parser.add_argument("--max-train-samples", type=int, default=MAX_TRAIN_SAMPLES)
    parser.add_argument("--max-val-samples", type=int, default=MAX_VAL_SAMPLES)
    parser.add_argument("--max-test-samples", type=int, default=MAX_TEST_SAMPLES)
    return parser


def add_model_args(parser):
    parser.add_argument("--model-type", type=str, default=MODEL_TYPE, choices=["wyckoff_transformer"])
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--discrete-schedule", type=str, default=DISCRETE_SCHEDULE)
    parser.add_argument("--continuous-schedule", type=str, default=CONTINUOUS_SCHEDULE)
    parser.add_argument("--beta-start", type=float, default=BETA_START)
    parser.add_argument("--beta-end", type=float, default=BETA_END)
    parser.add_argument("--discrete-loss-weight", type=float, default=DISCRETE_LOSS_WEIGHT)
    parser.add_argument("--continuous-loss-weight", type=float, default=CONTINUOUS_LOSS_WEIGHT)
    parser.add_argument("--lattice-loss-weight", type=float, default=LATTICE_LOSS_WEIGHT)
    parser.add_argument("--orbit-count-loss-weight", type=float, default=ORBIT_COUNT_LOSS_WEIGHT)
    parser.add_argument("--multiplicity-count-loss-weight", type=float, default=MULTIPLICITY_COUNT_LOSS_WEIGHT)
    parser.add_argument("--use-count-losses", dest="use_count_losses", action="store_true")
    parser.add_argument("--no-count-losses", dest="use_count_losses", action="store_false")
    parser.set_defaults(use_count_losses=USE_COUNT_LOSSES)
    parser.add_argument("--count-loss-orbit-margin", type=float, default=COUNT_LOSS_ORBIT_MARGIN)
    parser.add_argument("--count-loss-atom-margin", type=float, default=COUNT_LOSS_ATOM_MARGIN)
    parser.add_argument("--count-loss-smooth-tau", type=float, default=COUNT_LOSS_SMOOTH_TAU)
    parser.add_argument("--lambda-contrastive-con", type=float, default=LAMBDA_CONTRASTIVE_CON)
    parser.add_argument("--lambda-contrastive-dis", type=float, default=LAMBDA_CONTRASTIVE_DIS)
    parser.add_argument("--contrastive-margin", type=float, default=CONTRASTIVE_MARGIN)
    parser.add_argument("--contrastive-use-x0", dest="contrastive_use_x0", action="store_true",
                        help="Use reconstructed x0 errors for continuous contrastive learning.")
    parser.add_argument("--no-contrastive-use-x0", dest="contrastive_use_x0", action="store_false")
    parser.set_defaults(contrastive_use_x0=CONTRASTIVE_USE_X0)
    parser.add_argument("--max-wyckoff-sites", type=int, default=MAX_WYCKOFF_SITES)
    parser.add_argument("--max-inf-occ", type=int, default=DEFAULT_MAX_INF_OCC)
    parser.add_argument("--transformer-d-model", type=int, default=TRANSFORMER_D_MODEL)
    parser.add_argument("--transformer-nhead", type=int, default=TRANSFORMER_NHEAD)
    parser.add_argument("--transformer-layers", type=int, default=TRANSFORMER_LAYERS)
    parser.add_argument("--transformer-ff-dim", type=int, default=TRANSFORMER_FF_DIM)
    parser.add_argument("--transformer-dropout", type=float, default=TRANSFORMER_DROPOUT)
    parser.add_argument(
        "--transformer-denoiser-mode",
        type=str,
        default=TRANSFORMER_DENOISER_MODE,
        choices=["shared", "separate"],
    )
    return parser


def add_train_args(parser):
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--val-every", type=int, default=VAL_EVERY)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--resume", type=str, default=RESUME)
    parser.add_argument("--resume-best", dest="resume_best", action="store_true")
    parser.add_argument("--no-resume-best", dest="resume_best", action="store_false")
    parser.set_defaults(resume_best=RESUME_BEST)
    parser.add_argument("--log-file", type=str, default=str(TRAIN_LOG_FILE))
    parser.add_argument("--no-log-file", action="store_true")
    return parser


def add_test_args(parser):
    parser.add_argument("--checkpoint-path", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument("--eval-split", type=str, default=EVAL_SPLIT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-sample-batches", type=int, default=NUM_SAMPLE_BATCHES)
    parser.add_argument("--sample-batch-size", type=int, default=SAMPLE_BATCH_SIZE)
    parser.add_argument("--spacegroup-template-path", type=str, default=str(SPACEGROUP_TEMPLATE_PATH))
    parser.add_argument("--sample-space-group", type=int, default=SAMPLE_SPACE_GROUP)
    parser.add_argument(
        "--sample-space-groups",
        type=int,
        nargs="*",
        default=list(SAMPLE_SPACE_GROUPS),
        metavar="SPG",
        help="Sample each generated structure uniformly from these space-group numbers.",
    )
    parser.add_argument("--save-samples", dest="save_samples", action="store_true")
    parser.add_argument("--no-save-samples", dest="save_samples", action="store_false")
    parser.set_defaults(save_samples=SAVE_SAMPLES)
    parser.add_argument("--save-cifs", dest="save_cifs", action="store_true")
    parser.add_argument("--no-save-cifs", dest="save_cifs", action="store_false")
    parser.set_defaults(save_cifs=SAVE_CIFS)
    parser.add_argument("--num-save-cifs", type=int, default=NUM_SAVE_CIFS)
    parser.add_argument("--log-file", type=str, default=str(TEST_LOG_FILE))
    parser.add_argument("--no-log-file", action="store_true")
    return parser
