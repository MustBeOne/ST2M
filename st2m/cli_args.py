from __future__ import annotations

from st2m.utils import config as cfg


def add_common_args(parser):
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    parser.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=cfg.PIN_MEMORY)
    parser.add_argument("--train-dir", type=str, default=str(cfg.TRAIN_DIR))
    parser.add_argument("--val-dir", type=str, default=str(cfg.VAL_DIR))
    parser.add_argument("--test-dir", type=str, default=str(cfg.TEST_DIR))
    parser.add_argument("--output-dir", type=str, default=str(cfg.OUTPUT_DIR))
    parser.add_argument("--checkpoint-dir", type=str, default=str(cfg.CHECKPOINT_DIR))
    parser.add_argument("--sample-dir", type=str, default=str(cfg.SAMPLE_DIR))
    parser.add_argument("--lattice-dim", type=int, default=cfg.LATTICE_DIM)
    parser.add_argument("--num-discrete-classes", type=int, default=cfg.NUM_DISCRETE_CLASSES)
    parser.add_argument("--vacancy-class", type=int, default=cfg.VACANCY_CLASS)
    parser.add_argument("--max-train-samples", type=int, default=cfg.MAX_TRAIN_SAMPLES)
    parser.add_argument("--max-val-samples", type=int, default=cfg.MAX_VAL_SAMPLES)
    parser.add_argument("--max-test-samples", type=int, default=cfg.MAX_TEST_SAMPLES)
    return parser


def add_model_args(parser):
    parser.add_argument("--model-type", type=str, default=cfg.MODEL_TYPE, choices=["wyckoff_transformer"])
    parser.add_argument("--timesteps", type=int, default=cfg.TIMESTEPS)
    parser.add_argument("--discrete-schedule", type=str, default=cfg.DISCRETE_SCHEDULE)
    parser.add_argument("--continuous-schedule", type=str, default=cfg.CONTINUOUS_SCHEDULE)
    parser.add_argument("--beta-start", type=float, default=cfg.BETA_START)
    parser.add_argument("--beta-end", type=float, default=cfg.BETA_END)
    parser.add_argument("--discrete-loss-weight", type=float, default=cfg.DISCRETE_LOSS_WEIGHT)
    parser.add_argument("--continuous-loss-weight", type=float, default=cfg.CONTINUOUS_LOSS_WEIGHT)
    parser.add_argument("--lattice-loss-weight", type=float, default=cfg.LATTICE_LOSS_WEIGHT)
    parser.add_argument("--orbit-count-loss-weight", type=float, default=cfg.ORBIT_COUNT_LOSS_WEIGHT)
    parser.add_argument("--multiplicity-count-loss-weight", type=float, default=cfg.MULTIPLICITY_COUNT_LOSS_WEIGHT)
    parser.add_argument("--use-count-losses", dest="use_count_losses", action="store_true")
    parser.add_argument("--no-count-losses", dest="use_count_losses", action="store_false")
    parser.set_defaults(use_count_losses=cfg.USE_COUNT_LOSSES)
    parser.add_argument("--count-loss-orbit-margin", type=float, default=cfg.COUNT_LOSS_ORBIT_MARGIN)
    parser.add_argument("--count-loss-atom-margin", type=float, default=cfg.COUNT_LOSS_ATOM_MARGIN)
    parser.add_argument("--count-loss-smooth-tau", type=float, default=cfg.COUNT_LOSS_SMOOTH_TAU)
    parser.add_argument("--lambda-contrastive-con", type=float, default=cfg.LAMBDA_CONTRASTIVE_CON)
    parser.add_argument("--lambda-contrastive-dis", type=float, default=cfg.LAMBDA_CONTRASTIVE_DIS)
    parser.add_argument("--contrastive-margin", type=float, default=cfg.CONTRASTIVE_MARGIN)
    parser.add_argument("--contrastive-use-x0", dest="contrastive_use_x0", action="store_true",
                        help="Use reconstructed x0 errors for continuous contrastive learning.")
    parser.add_argument("--no-contrastive-use-x0", dest="contrastive_use_x0", action="store_false")
    parser.set_defaults(contrastive_use_x0=cfg.CONTRASTIVE_USE_X0)
    parser.add_argument("--max-wyckoff-sites", type=int, default=cfg.MAX_WYCKOFF_SITES)
    parser.add_argument("--max-inf-occ", type=int, default=cfg.DEFAULT_MAX_INF_OCC)
    parser.add_argument("--transformer-d-model", type=int, default=cfg.TRANSFORMER_D_MODEL)
    parser.add_argument("--transformer-nhead", type=int, default=cfg.TRANSFORMER_NHEAD)
    parser.add_argument("--transformer-layers", type=int, default=cfg.TRANSFORMER_LAYERS)
    parser.add_argument("--transformer-ff-dim", type=int, default=cfg.TRANSFORMER_FF_DIM)
    parser.add_argument("--transformer-dropout", type=float, default=cfg.TRANSFORMER_DROPOUT)
    parser.add_argument(
        "--transformer-denoiser-mode",
        type=str,
        default=cfg.TRANSFORMER_DENOISER_MODE,
        choices=["shared", "separate"],
    )
    return parser


def add_train_args(parser):
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=cfg.GRAD_CLIP)
    parser.add_argument("--log-every", type=int, default=cfg.LOG_EVERY)
    parser.add_argument("--val-every", type=int, default=cfg.VAL_EVERY)
    parser.add_argument("--save-every", type=int, default=cfg.SAVE_EVERY)
    parser.add_argument("--resume", type=str, default=cfg.RESUME)
    parser.add_argument("--resume-best", dest="resume_best", action="store_true")
    parser.add_argument("--no-resume-best", dest="resume_best", action="store_false")
    parser.set_defaults(resume_best=cfg.RESUME_BEST)
    parser.add_argument("--log-file", type=str, default=str(cfg.TRAIN_LOG_FILE))
    parser.add_argument("--no-log-file", action="store_true")
    return parser


def add_test_args(parser):
    parser.add_argument("--checkpoint-path", type=str, default=str(cfg.CHECKPOINT_PATH))
    parser.add_argument("--eval-split", type=str, default=cfg.EVAL_SPLIT)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--num-sample-batches", type=int, default=cfg.NUM_SAMPLE_BATCHES)
    parser.add_argument("--sample-batch-size", type=int, default=cfg.SAMPLE_BATCH_SIZE)
    parser.add_argument("--spacegroup-template-path", type=str, default=str(cfg.SPACEGROUP_TEMPLATE_PATH))
    parser.add_argument("--sample-space-group", type=int, default=cfg.SAMPLE_SPACE_GROUP)
    parser.add_argument(
        "--sample-space-groups",
        type=int,
        nargs="*",
        default=list(cfg.SAMPLE_SPACE_GROUPS),
        metavar="SPG",
        help="Sample each generated structure uniformly from these space-group numbers.",
    )
    parser.add_argument("--save-samples", dest="save_samples", action="store_true")
    parser.add_argument("--no-save-samples", dest="save_samples", action="store_false")
    parser.set_defaults(save_samples=cfg.SAVE_SAMPLES)
    parser.add_argument("--save-cifs", dest="save_cifs", action="store_true")
    parser.add_argument("--no-save-cifs", dest="save_cifs", action="store_false")
    parser.set_defaults(save_cifs=cfg.SAVE_CIFS)
    parser.add_argument("--num-save-cifs", type=int, default=cfg.NUM_SAVE_CIFS)
    parser.add_argument("--log-file", type=str, default=str(cfg.TEST_LOG_FILE))
    parser.add_argument("--no-log-file", action="store_true")
    return parser
