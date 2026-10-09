from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any
from tqdm import tqdm
import numpy as np


TARGET_SPACE_GROUPS = (
    225,
    12,
    139,
    62,
    194,
    166,
    63,
    221,
    2,
    123,
    14,
    164,
    216,
    15,
    129,
    1,
    189,
    71,
    38,
    8,
    148,
)
TEMPLATES_PER_SPACE_GROUP = 50
RANDOM_SEED = None
MAX_SAMPLING_ATTEMPTS = 1_000_000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data" / "mp20" / "processed" / "train"
FREQUENCY_FILE = PROJECT_ROOT / "outputs" / "metric" / "sg_orbit_time_dist.json"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "postproc" / "new_orbit_templates.json"


def _load_npy_record(path: Path) -> dict[str, Any]:
    raw = np.load(path, allow_pickle=True)
    record = raw.item() if isinstance(raw, np.ndarray) and raw.shape == () else raw
    if not isinstance(record, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(record).__name__}.")
    return record


def canonical_template(letters: Any) -> tuple[str, ...]:
    """Represent a Wyckoff template as an a-to-z ordered multiset."""
    return tuple(sorted(str(letter) for letter in letters))


def load_training_templates(
    train_dir: Path,
    target_space_groups: set[int],
) -> dict[int, set[tuple[str, ...]]]:
    templates = {spg: set() for spg in target_space_groups}
    paths = sorted(train_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No training .npy files found under {train_dir}.")

    for path in paths:
        record = _load_npy_record(path)
        spg = int(record["spg_number"])
        if spg not in templates:
            continue
        letters = canonical_template(record["wyckoff_letters"])
        if letters:
            templates[spg].add(letters)
    return templates


def load_occurrence_distributions(
    frequency_file: Path,
    target_space_groups: set[int],
) -> dict[int, dict[str, list[tuple[int, int]]]]:
    if not frequency_file.exists():
        raise FileNotFoundError(f"Frequency file not found: {frequency_file}")
    raw = json.loads(frequency_file.read_text(encoding="utf-8"))
    distributions: dict[int, dict[str, list[tuple[int, int]]]] = {}

    for spg in sorted(target_space_groups):
        group = raw.get(str(spg))
        if not isinstance(group, dict) or not group:
            raise KeyError(f"Space group {spg} is missing from {frequency_file}.")

        site_distributions: dict[str, list[tuple[int, int]]] = {}
        for letter in sorted(group):
            frequencies = [
                (int(count), int(frequency))
                for count, frequency in group[letter].items()
                if int(frequency) > 0
            ]
            frequencies.sort()
            if not frequencies:
                raise ValueError(f"Space group {spg}, site {letter} has no positive frequency.")
            if any(count < 0 for count, _ in frequencies):
                raise ValueError(f"Space group {spg}, site {letter} has a negative occurrence count.")
            site_distributions[str(letter)] = frequencies
        distributions[spg] = site_distributions
    return distributions


def count_unseen_support(
    site_distributions: dict[str, list[tuple[int, int]]],
    seen_templates: set[tuple[str, ...]],
) -> tuple[int, int]:
    letters = sorted(site_distributions)
    choices = [site_distributions[letter] for letter in letters]
    support_size = math.prod(len(site_choices) for site_choices in choices)
    unseen_templates: set[tuple[str, ...]] = set()

    for combination in itertools.product(*choices):
        counts = tuple(count for count, _ in combination)
        if not any(counts):
            continue
        template = tuple(
            letter
            for letter, count in zip(letters, counts)
            for _ in range(count)
        )
        if template not in seen_templates:
            unseen_templates.add(template)
    return len(unseen_templates), support_size


def sample_unseen_templates(
    site_distributions: dict[str, list[tuple[int, int]]],
    seen_templates: set[tuple[str, ...]],
    sample_count: int,
    rng: random.Random,
) -> tuple[list[tuple[str, ...]], int, int, int]:
    """Repeatedly draw each site's count and reject seen templates."""
    letters = sorted(site_distributions)
    available_unseen, support_size = count_unseen_support(
        site_distributions,
        seen_templates,
    )
    target_count = min(sample_count, available_unseen)
    selected: list[tuple[str, ...]] = []
    selected_set: set[tuple[str, ...]] = set()
    attempts = 0
    bar=tqdm(total=MAX_SAMPLING_ATTEMPTS)
    bar.set_description(f'Sampled:0/{target_count}')
    while len(selected) < target_count and attempts < MAX_SAMPLING_ATTEMPTS:
        attempts += 1
        bar.update(1)
        counts = []
        for letter in letters:
            count_frequency_pairs = site_distributions[letter]
            possible_counts = [count for count, _ in count_frequency_pairs]
            frequencies = [frequency for _, frequency in count_frequency_pairs]
            counts.append(rng.choices(possible_counts, weights=frequencies, k=1)[0])

        if not any(counts):
            continue
        template = tuple(
            letter
            for letter, count in zip(letters, counts)
            for _ in range(count)
        )
        if template in seen_templates or template in selected_set:
            continue
        bar.set_description(f'Sampled:{len(selected)}/{target_count}')
        selected.append(template)
        selected_set.add(template)

    return selected, available_unseen, support_size, attempts


def build_output(
    target_space_groups: tuple[int, ...],
    templates_per_space_group: int,
    seed: int | None,
    train_dir: Path,
    frequency_file: Path,
) -> dict[str, Any]:
    target_set = set(target_space_groups)
    training_templates = load_training_templates(train_dir, target_set)
    distributions = load_occurrence_distributions(frequency_file, target_set)
    rng = random.Random(seed)
    generated: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, dict[str, Any]] = {}

    for spg in target_space_groups:
        if spg==62:
            a=0
        templates, available_unseen, support_size, attempts = sample_unseen_templates(
            distributions[spg],
            training_templates[spg],
            templates_per_space_group,
            rng,
        )
        records = [
            {
                "wyckoff_letters": list(template),
                "orbit_count": len(template),
            }
            for template in templates
        ]
        generated[str(spg)] = records
        summary[str(spg)] = {
            "requested": templates_per_space_group,
            "generated": len(records),
            "available_unseen_unique": available_unseen,
            "training_unique": len(training_templates[spg]),
            "cartesian_support_size": support_size,
            "sampling_attempts": attempts,
            "complete": len(records) == templates_per_space_group,
        }

    return {
        "metadata": {
            "seed": seed,
            "sampling": "independent random.choices draw per site followed by rejection sampling",
            "novelty_reference": str(train_dir.resolve()),
            "frequency_source": str(frequency_file.resolve()),
            "space_groups": list(target_space_groups),
            "templates_per_space_group": templates_per_space_group,
        },
        "space_groups": generated,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate occupied-orbit templates that do not occur in the training set."
    )
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--frequency-file", type=Path, default=FREQUENCY_FILE)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--templates-per-space-group", type=int, default=TEMPLATES_PER_SPACE_GROUP)
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Optional random seed. By default each run uses fresh system randomness.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.templates_per_space_group <= 0:
        raise ValueError("--templates-per-space-group must be positive.")

    output = build_output(
        target_space_groups=TARGET_SPACE_GROUPS,
        templates_per_space_group=args.templates_per_space_group,
        seed=args.seed,
        train_dir=args.train_dir,
        frequency_file=args.frequency_file,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved new templates to {args.output_file}")
    for spg in TARGET_SPACE_GROUPS:
        status = output["summary"][str(spg)]
        message = f"SG {spg}: {status['generated']}/{status['requested']} unique unseen templates"
        if not status["complete"]:
            message += f" (only {status['available_unseen_unique']} exist in the sampled support)"
        print(message)


if __name__ == "__main__":
    main()
