from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SEED = 42
TARGET_SIZES = {
    "train": 480,
    "validation": 60,
    "test": 60,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Neispravan JSON u {path} na liniji {line_no}: {exc}"
                ) from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_user_text(example: dict[str, Any]) -> str:
    for message in example.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def normalize_question(text: str) -> str:
    """Normalizuje pitanje za stabilno prepoznavanje porodice pitanja."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\d+", "<num>", text)
    text = re.sub(r"[„“”\"'`´]", "", text)
    text = re.sub(r"[^\w<>]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalized_question_family(example: dict[str, Any]) -> str:
    """
    T11 `type` predstavlja šablonsku/namjensku porodicu pitanja.

    Za primjere vezane za destinaciju koristimo `type` kao kanonsku
    porodicu. Time eventualne parafraze istog tipa za istu destinaciju
    ostaju u istom splitu.

    Za primjere bez destination_id (npr. out_of_domain) dodajemo
    normalizovan tekst pitanja da svi takvi primjeri ne postanu jedna
    ogromna grupa.
    """
    example_type = str(example.get("type", "")).strip()
    destination_ids = tuple(sorted(example.get("destination_ids") or []))

    if destination_ids:
        return example_type

    return f"{example_type}::{normalize_question(first_user_text(example))}"


def group_key(example: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    destination_ids = tuple(sorted(example.get("destination_ids") or []))
    family = normalized_question_family(example)
    return destination_ids, family


def build_groups(
    examples: list[dict[str, Any]],
) -> dict[tuple[tuple[str, ...], str], list[dict[str, Any]]]:
    groups: dict[tuple[tuple[str, ...], str], list[dict[str, Any]]] = defaultdict(list)

    for example in examples:
        groups[group_key(example)].append(example)

    # Jedna porodica mora pripadati samo jednom type-u da bi stratifikacija
    # po type-u bila jednoznačna.
    for key, members in groups.items():
        member_types = {str(x.get("type", "")) for x in members}
        if len(member_types) != 1:
            raise ValueError(
                f"Grupa {key} sadrži više type vrijednosti: {sorted(member_types)}"
            )

    return dict(groups)


def allocate_holdout_targets(
    counts_by_type: Counter[str],
    total_holdout: int,
) -> dict[str, int]:
    """
    Largest-remainder raspodjela 10% holdout-a po type-u.
    Svaki type dobija najmanje 1 primjer u holdout splitu.
    """
    total_examples = sum(counts_by_type.values())
    ideal = {
        t: counts_by_type[t] * total_holdout / total_examples
        for t in counts_by_type
    }

    targets = {
        t: max(1, math.floor(ideal[t]))
        for t in counts_by_type
    }

    current = sum(targets.values())

    if current > total_holdout:
        # U ovom datasetu se ne očekuje, ali ostavljamo jasnu grešku.
        raise ValueError(
            "Previše type kategorija za traženu veličinu holdout splita."
        )

    remainders = sorted(
        counts_by_type,
        key=lambda t: (ideal[t] - math.floor(ideal[t]), counts_by_type[t], t),
        reverse=True,
    )

    idx = 0
    while current < total_holdout:
        t = remainders[idx % len(remainders)]
        # Train mora zadržati najmanje 1 primjer čak i nakon val+test.
        if counts_by_type[t] - 2 * (targets[t] + 1) >= 1:
            targets[t] += 1
            current += 1
        idx += 1

        if idx > 10000:
            raise RuntimeError("Nije moguće izračunati holdout targete.")

    return targets


def choose_groups_for_exact_count(
    groups: list[tuple[tuple[tuple[str, ...], str], list[dict[str, Any]]]],
    target: int,
) -> set[tuple[tuple[str, ...], str]]:
    """
    Deterministički subset-sum nad veličinama grupa.

    Vraća group_key vrijednosti čiji zbir broja primjera daje tačno target.
    """
    dp: dict[int, list[int]] = {0: []}

    for idx, (_, members) in enumerate(groups):
        size = len(members)
        snapshot = list(dp.items())

        for subtotal, chosen in snapshot:
            new_total = subtotal + size
            if new_total > target or new_total in dp:
                continue
            dp[new_total] = chosen + [idx]

        if target in dp:
            break

    if target not in dp:
        sizes = [len(members) for _, members in groups]
        raise ValueError(
            f"Nije moguće složiti tačno {target} primjera iz group veličina {sizes}."
        )

    return {groups[idx][0] for idx in dp[target]}


def split_examples(
    examples: list[dict[str, Any]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if len(examples) != 600:
        raise ValueError(
            f"T13 očekuje tačno 600 finalnih primjera, pronađeno: {len(examples)}"
        )

    example_ids = [str(x.get("example_id", "")) for x in examples]
    if any(not x for x in example_ids):
        raise ValueError("Postoji primjer bez example_id.")
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("example_id se ponavlja u all_examples.jsonl.")

    counts_by_type = Counter(str(x.get("type", "")) for x in examples)
    if any(not t for t in counts_by_type):
        raise ValueError("Postoji primjer bez type vrijednosti.")

    # 60 validation i 60 test; train je ostatak 480.
    holdout_targets = allocate_holdout_targets(
        counts_by_type=counts_by_type,
        total_holdout=TARGET_SIZES["validation"],
    )

    groups = build_groups(examples)

    groups_by_type: dict[
        str,
        list[tuple[tuple[tuple[str, ...], str], list[dict[str, Any]]]],
    ] = defaultdict(list)

    for key, members in groups.items():
        example_type = str(members[0]["type"])
        groups_by_type[example_type].append((key, members))

    rng = random.Random(seed)

    split_group_keys: dict[str, set[tuple[tuple[str, ...], str]]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    # Stratifikacija se radi po type-u, ali cijele question-family grupe
    # uvijek ostaju zajedno.
    for example_type in sorted(groups_by_type):
        type_groups = list(groups_by_type[example_type])

        # Stabilan redoslijed prije seedovanog shuffle-a.
        type_groups.sort(key=lambda item: repr(item[0]))
        rng.shuffle(type_groups)

        val_target = holdout_targets[example_type]
        test_target = holdout_targets[example_type]

        validation_keys = choose_groups_for_exact_count(
            type_groups,
            val_target,
        )

        remaining_after_val = [
            item for item in type_groups
            if item[0] not in validation_keys
        ]

        test_keys = choose_groups_for_exact_count(
            remaining_after_val,
            test_target,
        )

        train_keys = {
            key for key, _ in remaining_after_val
            if key not in test_keys
        }

        split_group_keys["validation"].update(validation_keys)
        split_group_keys["test"].update(test_keys)
        split_group_keys["train"].update(train_keys)

    # Materijalizuj primjere po splitu u stabilnom originalnom redoslijedu.
    split_by_id: dict[str, str] = {}
    for split_name, keys in split_group_keys.items():
        for key in keys:
            for example in groups[key]:
                eid = str(example["example_id"])
                if eid in split_by_id:
                    raise AssertionError(
                        f"Primjer {eid} dodijeljen je u više splitova."
                    )
                split_by_id[eid] = split_name

    if len(split_by_id) != len(examples):
        missing = sorted(set(example_ids) - set(split_by_id))
        raise AssertionError(
            f"Nisu svi primjeri dodijeljeni splitu. Nedostaju: {missing[:10]}"
        )

    result = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for example in examples:
        result[split_by_id[str(example["example_id"])]].append(example)

    validate_split(result, groups)

    return result


def validate_split(
    splits: dict[str, list[dict[str, Any]]],
    groups: dict[tuple[tuple[str, ...], str], list[dict[str, Any]]],
) -> None:
    # Tačni brojevi 480/60/60.
    for split_name, expected_size in TARGET_SIZES.items():
        actual_size = len(splits[split_name])
        if actual_size != expected_size:
            raise AssertionError(
                f"{split_name}: očekivano {expected_size}, dobijeno {actual_size}"
            )

    # Nema example_id overlap-a.
    id_sets = {
        split_name: {str(x["example_id"]) for x in rows}
        for split_name, rows in splits.items()
    }

    if id_sets["train"] & id_sets["validation"]:
        raise AssertionError("example_id overlap između train i validation.")
    if id_sets["train"] & id_sets["test"]:
        raise AssertionError("example_id overlap između train i test.")
    if id_sets["validation"] & id_sets["test"]:
        raise AssertionError("example_id overlap između validation i test.")

    # Svi tipovi pitanja moraju biti u sva tri splita.
    all_types = {
        str(x["type"])
        for rows in splits.values()
        for x in rows
    }

    for split_name, rows in splits.items():
        present_types = {str(x["type"]) for x in rows}
        missing_types = sorted(all_types - present_types)
        if missing_types:
            raise AssertionError(
                f"{split_name} nema sljedeće type vrijednosti: {missing_types}"
            )

    # Cijela (destination_id + normalized_question_family) grupa mora
    # završiti u samo jednom splitu.
    split_of_id = {}
    for split_name, rows in splits.items():
        for row in rows:
            split_of_id[str(row["example_id"])] = split_name

    for key, members in groups.items():
        member_splits = {
            split_of_id[str(member["example_id"])]
            for member in members
        }
        if len(member_splits) != 1:
            raise AssertionError(
                f"Question-family grupa {key} je podijeljena između splitova: "
                f"{sorted(member_splits)}"
            )


def build_stats_rows(
    splits: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    all_types = sorted({
        str(x["type"])
        for split_rows in splits.values()
        for x in split_rows
    })

    for example_type in all_types:
        values = {
            split_name: sum(
                1 for x in split_rows
                if str(x["type"]) == example_type
            )
            for split_name, split_rows in splits.items()
        }
        rows.append({
            "dimension": "type",
            "key": example_type,
            "train": values["train"],
            "validation": values["validation"],
            "test": values["test"],
            "total": sum(values.values()),
        })

    destination_ids = sorted({
        destination_id
        for split_rows in splits.values()
        for x in split_rows
        for destination_id in (x.get("destination_ids") or [])
    })

    for destination_id in destination_ids:
        values = {
            split_name: sum(
                1 for x in split_rows
                if destination_id in (x.get("destination_ids") or [])
            )
            for split_name, split_rows in splits.items()
        }
        rows.append({
            "dimension": "destination_id",
            "key": destination_id,
            "train": values["train"],
            "validation": values["validation"],
            "test": values["test"],
            "total": sum(values.values()),
        })

    # Primjeri bez destinacije, npr. out_of_domain.
    no_destination = {
        split_name: sum(
            1 for x in split_rows
            if not (x.get("destination_ids") or [])
        )
        for split_name, split_rows in splits.items()
    }
    rows.append({
        "dimension": "destination_id",
        "key": "__NO_DESTINATION__",
        "train": no_destination["train"],
        "validation": no_destination["validation"],
        "test": no_destination["test"],
        "total": sum(no_destination.values()),
    })

    return rows


def write_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dimension",
        "key",
        "train",
        "validation",
        "test",
        "total",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "T13: deterministička i grupno-stratifikovana podjela "
            "600 finalnih SFT primjera na 480/60/60."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/all_examples.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/sft"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/reports/split_stats.csv"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    args = parser.parse_args()

    examples = read_jsonl(args.input)
    groups = build_groups(examples)
    splits = split_examples(examples, seed=args.seed)

    write_jsonl(args.output_dir / "train.jsonl", splits["train"])
    write_jsonl(args.output_dir / "validation.jsonl", splits["validation"])
    write_jsonl(args.output_dir / "test.jsonl", splits["test"])

    stats_rows = build_stats_rows(splits)
    write_stats_csv(args.report, stats_rows)

    print("T13 split završen.")
    print(f"seed={args.seed}")
    print(f"train={len(splits['train'])}")
    print(f"validation={len(splits['validation'])}")
    print(f"test={len(splits['test'])}")
    print(f"Question-family grupa: {len(groups)}")
    print(f"Sačuvan izvještaj: {args.report}")


if __name__ == "__main__":
    main()
