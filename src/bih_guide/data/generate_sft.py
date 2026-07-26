"""Generate deterministic supervised fine-tuning candidates from the RAG knowledge base."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

BASE_DIR = Path(__file__).resolve().parents[3]
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "processed" / "knowledge_base.jsonl"
DESTINATIONS_PATH = BASE_DIR / "data" / "destination_registry.csv"
TEMPLATES_PATH = BASE_DIR / "configs" / "sft_templates.yaml"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "sft_candidates.jsonl"
STATS_PATH = BASE_DIR / "artifacts" / "reports" / "sft_candidate_stats.json"

DESTINATION_TYPES = {
    "destination_description",
    "destination_attractions",
    "destination_nature",
    "destination_activity",
    "one_day_plan",
    "multi_day_plan",
    "interest_recommendation",
    "route_connection",
}
NON_FACTUAL_TYPES = {"clarification", "out_of_domain"}
REQUIRED_FIELDS = {
    "example_id",
    "type",
    "destination_ids",
    "chunk_ids",
    "source_ids",
    "reviewed",
    "review_status",
    "question_family",
    "messages",
}
ROLE_SEQUENCE_MULTI_TURN = ["system", "user", "assistant", "user", "assistant"]
INTERESTS = ["prirodu", "istoriju", "kulturu", "aktivnosti na otvorenom"]
SEASONS = ["proljeće", "ljeto", "jesen", "zima"]
TRANSPORTS = ["automobilom", "javnim prevozom", "kombinovano"]
DYNAMIC_TOPICS = [
    ("cijena ulaznice", "tačnu cijenu ulaznice"),
    ("radno vrijeme", "tačno radno vrijeme"),
    ("red vožnje", "aktuelni red vožnje"),
    ("datum događaja", "tačan datum događaja"),
    ("dostupnost smještaja", "trenutnu dostupnost smještaja"),
]
OFF_DOMAIN_PROMPTS = [
    "Objasni mi kako radi kvadratna jednačina.",
    "Napiši kratak program za sortiranje brojeva.",
    "Ko je napisao roman Ana Karenjina?",
    "Kako nastaje duga?",
    "Koja su pravila košarke?",
    "Objasni fotosintezu jednostavnim riječima.",
    "Kako da napravim prezentaciju o svemiru?",
    "Koja je razlika između RAM-a i SSD-a?",
    "Prepričaj mi teoriju evolucije.",
    "Kako funkcioniše električno kolo?",
    "Napiši sastav o prijateljstvu.",
    "Šta je Pitagorina teorema?",
    "Kako se računa površina kruga?",
    "Objasni mi binarni brojni sistem.",
    "Koje su glavne osobine sisara?",
    "Kako nastaju zemljotresi?",
    "Šta je algoritam?",
    "Objasni ulogu mitohondrija.",
    "Kako se piše formalno poslovno pismo?",
    "Koja je razlika između mase i težine?",
    "Napravi plan učenja matematike za sedmicu.",
    "Objasni šta je baza podataka.",
    "Koje su planete Sunčevog sistema?",
    "Kako se određuje brzina tijela?",
    "Šta je metafora u književnosti?",
    "Objasni osnovne principe programiranja.",
    "Koji su uzroci smjene godišnjih doba?",
    "Kako radi internet pretraživač?",
    "Šta je hemijska reakcija?",
    "Objasni razliku između virusa i bakterija.",
    "Kako se pravi tabela u HTML-u?",
    "Koje su funkcije korijena biljke?",
    "Šta znači pojam demokratija?",
    "Kako se izračunava prosjek ocjena?",
    "Objasni šta je gravitacija.",
    "Koje su vrste rečenica u gramatici?",
    "Kako funkcioniše računar?",
    "Šta je obnovljivi izvor energije?",
    "Objasni kruženje vode u prirodi.",
    "Kako se piše bibliografija?",
    "Koja je razlika između hardvera i softvera?",
    "Šta je ćelijska membrana?",
    "Objasni pojam vjerovatnoće.",
    "Kako se formira oblak?",
    "Koje su osnovne jedinice SI sistema?",
    "Šta je operativni sistem?",
    "Objasni šta je književni lik.",
    "Kako se računa zapremina kocke?",
    "Koja je uloga DNK?",
    "Objasni šta je mašinsko učenje.",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per non-empty line."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} in {path} must be a JSON object.")
            rows.append(row)
    return rows


def read_destinations(path: Path) -> list[dict[str, str]]:
    """Read the ordered destination registry."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"destination_id", "name", "region"}
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}.")
        rows = [dict(row) for row in reader]

    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        destination_id = row["destination_id"].strip()
        name = row["name"].strip()
        region = row["region"].strip()
        if not destination_id or not name or not region:
            raise ValueError(f"Blank destination field on row {row_number} in {path}.")
        if destination_id in seen:
            raise ValueError(f"Duplicate destination_id {destination_id} in {path}.")
        seen.add(destination_id)
        row["destination_id"] = destination_id
        row["name"] = name
        row["region"] = region
    return rows


def load_templates(path: Path) -> dict[str, Any]:
    """Load and minimally validate the YAML candidate templates."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    required = {"system_prompt", "minimum_total_candidates", "additional_counts", "destination_templates"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"{path} is missing keys: {', '.join(sorted(missing))}.")

    templates = config["destination_templates"]
    if not isinstance(templates, list) or len(templates) < 8:
        raise ValueError("destination_templates must contain at least eight templates.")

    types = [template.get("type") for template in templates]
    if len(types) != len(set(types)):
        raise ValueError("destination template types must be unique.")
    if not DESTINATION_TYPES <= set(types):
        missing_types = DESTINATION_TYPES - set(types)
        raise ValueError(
            "Missing destination template types: " + ", ".join(sorted(missing_types))
        )
    return config


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def split_sentences(text: str) -> list[str]:
    """Split a chunk into sentence-like units and remove its generated heading."""
    normalized = normalize_space(text)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]
    if sentences and "—" in sentences[0]:
        sentences = sentences[1:]
    return sentences


def ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def choose_chunks(
    chunks: Sequence[dict[str, Any]],
    preferred_categories: Sequence[str],
    count: int = 1,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select chunks by category overlap with stable deterministic tie-breaking."""
    excluded_ids = excluded_ids or set()
    preferred = set(preferred_categories)
    ranked = sorted(
        (chunk for chunk in chunks if chunk["chunk_id"] not in excluded_ids),
        key=lambda chunk: (
            -len(preferred & set(chunk["categories"])),
            chunk["chunk_id"],
        ),
    )
    return ranked[:count]


def excerpt_from_chunks(
    chunks: Sequence[dict[str, Any]],
    max_sentences: int = 4,
    max_words: int = 120,
    offset: int = 0,
) -> str:
    """Build a sentence-bound excerpt without cutting a sentence mid-way."""
    sentence_pool: list[str] = []
    for chunk in chunks:
        sentence_pool.extend(split_sentences(chunk["text"]))

    if not sentence_pool:
        return ""

    rotated = sentence_pool[offset % len(sentence_pool):] + sentence_pool[: offset % len(sentence_pool)]
    chosen: list[str] = []
    word_count = 0
    for sentence in rotated:
        sentence_words = len(sentence.split())
        if chosen and word_count + sentence_words > max_words:
            break
        chosen.append(sentence)
        word_count += sentence_words
        if len(chosen) >= max_sentences:
            break
    return " ".join(chosen)


def source_ids_for(chunks: Sequence[dict[str, Any]]) -> list[str]:
    return ordered_unique(
        source_id
        for chunk in chunks
        for source_id in chunk["source_ids"]
    )


def chunk_ids_for(chunks: Sequence[dict[str, Any]]) -> list[str]:
    return [chunk["chunk_id"] for chunk in chunks]


def make_example(
    *,
    example_id: str,
    example_type: str,
    destination_ids: Sequence[str],
    chunks: Sequence[dict[str, Any]],
    question_family: str,
    messages: Sequence[dict[str, str]],
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "type": example_type,
        "destination_ids": list(destination_ids),
        "chunk_ids": chunk_ids_for(chunks),
        "source_ids": source_ids_for(chunks),
        "reviewed": False,
        "review_status": "pending",
        "question_family": question_family,
        "messages": list(messages),
    }


def destination_answer(
    template_type: str,
    destination_name: str,
    selected_chunks: Sequence[dict[str, Any]],
    interest: str,
    variant_index: int,
) -> str:
    """Create a grounded answer for one of the eight destination templates."""
    excerpt = excerpt_from_chunks(
        selected_chunks,
        max_sentences=4 if template_type not in {"one_day_plan", "multi_day_plan"} else 5,
        max_words=125,
        offset=variant_index,
    )

    if template_type == "destination_description":
        return (
            f"{destination_name} se može predstaviti kroz nekoliko provjerenih turističkih cjelina. "
            f"{excerpt} Za prvi obilazak izaberi nekoliko glavnih tačaka i ostavi dovoljno vremena "
            "za kretanje između njih."
        )
    if template_type == "destination_attractions":
        return (
            f"Za kulturno-istorijski obilazak destinacije {destination_name} izdvojio bih ove cjeline. "
            f"{excerpt} Redoslijed prilagodi lokaciji smještaja i važećem radnom vremenu ustanova."
        )
    if template_type == "destination_nature":
        return (
            f"Za ljubitelje prirode, {destination_name} nudi više mogućnosti. {excerpt} "
            "Prije aktivnosti na otvorenom provjeri vremenske uslove, stanje staza i lokalna pravila."
        )
    if template_type == "destination_activity":
        return (
            f"Boravak u destinaciji {destination_name} može se organizovati oko sljedećih aktivnosti. "
            f"{excerpt} Za zahtjevnije aktivnosti provjeri potrebnu opremu i mogućnost angažovanja vodiča."
        )
    if template_type == "one_day_plan":
        return (
            f"Jedan dan u destinaciji {destination_name} organizuj kao sažet obilazak bez previše tačaka. "
            f"{excerpt} Počni ranije, a tačan redoslijed prilagodi pristupu lokalitetima i aktuelnom radnom vremenu."
        )
    if template_type == "multi_day_plan":
        first = excerpt_from_chunks(selected_chunks[:1], max_sentences=3, max_words=75, offset=variant_index)
        second = excerpt_from_chunks(selected_chunks[1:] or selected_chunks[:1], max_sentences=3, max_words=75, offset=variant_index + 1)
        return (
            f"Prvi dan u destinaciji {destination_name} usmjeri na glavne gradske ili kulturne cjeline. "
            f"{first} Drugi dan ostavi za prirodu, aktivnosti ili obližnji izlet. {second} "
            "Prije puta provjeri vremenske uslove i promjenjive praktične informacije."
        )
    if template_type == "interest_recommendation":
        return (
            f"Destinacija {destination_name} može odgovarati posjetiocu sa interesovanjem za {interest}. "
            f"{excerpt} Preporuku prilagodi sezoni, kondiciji i raspoloživom vremenu."
        )
    if template_type == "route_connection":
        return (
            f"Destinaciju {destination_name} poveži samo sa mjestima i izletima koji su navedeni u provjerenom kontekstu. "
            f"{excerpt} Ne planiraj tijesan raspored bez provjere udaljenosti, puta i aktuelnog prevoza."
        )
    raise ValueError(f"Unsupported destination template type: {template_type}")


def generate_destination_examples(
    destinations: Sequence[dict[str, str]],
    chunks_by_destination: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate eight differently typed candidates for every destination."""
    examples: list[dict[str, Any]] = []
    system_prompt = config["system_prompt"]

    for destination_index, destination in enumerate(destinations):
        destination_id = destination["destination_id"]
        destination_name = destination["name"]
        chunks = chunks_by_destination[destination_id]

        for template_index, template in enumerate(config["destination_templates"]):
            template_type = template["type"]
            preferred = template.get("preferred_categories", [])
            selected_count = 2 if template_type in {"multi_day_plan", "route_connection"} else 1
            selected = choose_chunks(chunks, preferred, count=selected_count)
            interest = INTERESTS[(destination_index + template_index) % len(INTERESTS)]
            user = template["user"].format(
                destination_name=destination_name,
                interest=interest,
            )
            answer = destination_answer(
                template_type,
                destination_name,
                selected,
                interest,
                variant_index=destination_index + template_index,
            )
            examples.append(
                make_example(
                    example_id=f"SFT-DEST-{destination_id.replace('_', '-').upper()}-{template_index + 1:02d}",
                    example_type=template_type,
                    destination_ids=[destination_id],
                    chunks=selected,
                    question_family=f"{template_type}:{destination_id}",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": answer},
                    ],
                )
            )
    return examples


def destination_pairs(
    destinations: Sequence[dict[str, str]],
    count: int,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Build deterministic same-region pairs, preferring nearby registry positions."""
    by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    region_order: list[str] = []
    for destination in destinations:
        region = destination["region"]
        if region not in by_region:
            region_order.append(region)
        by_region[region].append(destination)

    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    for offset in range(1, 6):
        for region in region_order:
            group = by_region[region]
            if len(group) < 2:
                continue
            for index, first in enumerate(group):
                second_index = index + offset
                if second_index >= len(group):
                    continue
                second = group[second_index]
                key = tuple(sorted((first["destination_id"], second["destination_id"])))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((first, second))
                if len(pairs) >= count:
                    return pairs

    if len(pairs) < count:
        raise ValueError(f"Could create only {len(pairs)} unique same-region pairs; required {count}.")
    return pairs


def generate_multi_destination_plans(
    destinations: Sequence[dict[str, str]],
    chunks_by_destination: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    count = int(config["additional_counts"]["multi_destination_plan"])
    system_prompt = config["system_prompt"]
    pairs = destination_pairs(destinations, count)
    examples: list[dict[str, Any]] = []

    for index, (first, second) in enumerate(pairs, start=1):
        first_chunk = choose_chunks(
            chunks_by_destination[first["destination_id"]],
            ["route", "attraction", "description"],
        )[0]
        second_chunk = choose_chunks(
            chunks_by_destination[second["destination_id"]],
            ["route", "nature", "attraction"],
        )[0]
        transport = TRANSPORTS[(index - 1) % len(TRANSPORTS)]
        interest = INTERESTS[(index - 1) % len(INTERESTS)]
        user = (
            f"Napravi dvodnevni plan koji povezuje destinacije „{first['name']}” i „{second['name']}” "
            f"za putovanje {transport}, sa naglaskom na {interest}."
        )
        first_excerpt = excerpt_from_chunks([first_chunk], max_sentences=3, max_words=70, offset=index)
        second_excerpt = excerpt_from_chunks([second_chunk], max_sentences=3, max_words=70, offset=index + 1)
        answer = (
            f"Dan 1 — {first['name']}: {first_excerpt} "
            f"Dan 2 — {second['name']}: {second_excerpt} "
            "Redoslijed je okviran; prije polaska provjeri stanje puta ili aktuelne polaske javnog prevoza "
            "i ne oslanjaj se na neprovjerene satnice."
        )
        examples.append(
            make_example(
                example_id=f"SFT-ROUTE-{index:03d}",
                example_type="multi_destination_plan",
                destination_ids=[first["destination_id"], second["destination_id"]],
                chunks=[first_chunk, second_chunk],
                question_family=f"multi_destination:{first['destination_id']}:{second['destination_id']}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
            )
        )
    return examples


def generate_clarifications(
    destinations: Sequence[dict[str, str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    count = int(config["additional_counts"]["clarification"])
    system_prompt = config["system_prompt"]
    questions = [
        ("trajanje", "Koliko dana imaš na raspolaganju?"),
        ("prevoz", "Kojim prevozom planiraš putovati?"),
        ("interesovanje", "Šta te najviše zanima: priroda, istorija, kultura ili hrana?"),
        ("sezona", "U kojem godišnjem dobu planiraš putovanje?"),
        ("grupa", "Putuješ li samostalno, s prijateljima ili s porodicom?"),
    ]
    examples: list[dict[str, Any]] = []

    for index in range(count):
        destination = destinations[index % len(destinations)]
        dimension, assistant_question = questions[index % len(questions)]
        user = (
            f"Želim preporuku za posjetu destinaciji „{destination['name']}”, "
            f"ali još nisam odredio detalje plana broj {index + 1}."
        )
        examples.append(
            make_example(
                example_id=f"SFT-CLARIFY-{index + 1:03d}",
                example_type="clarification",
                destination_ids=[destination["destination_id"]],
                chunks=[],
                question_family=f"clarification:{dimension}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant_question},
                ],
            )
        )
    return examples


def generate_multi_turn(
    destinations: Sequence[dict[str, str]],
    chunks_by_destination: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    count = int(config["additional_counts"]["multi_turn"])
    system_prompt = config["system_prompt"]
    examples: list[dict[str, Any]] = []

    for index in range(count):
        destination = destinations[index % len(destinations)]
        destination_id = destination["destination_id"]
        destination_name = destination["name"]
        interest = INTERESTS[index % len(INTERESTS)]
        season = SEASONS[(index // len(INTERESTS)) % len(SEASONS)]
        transport = TRANSPORTS[index % len(TRANSPORTS)]
        days = 1 + (index % 3)
        selected = choose_chunks(
            chunks_by_destination[destination_id],
            ["route", "attraction", "nature", "activity"],
            count=2,
        )
        first_user = (
            f"Razmišljam o putovanju koje uključuje {destination_name}; "
            f"želim varijantu usmjerenu na {interest} u sezoni {season}."
        )
        day_phrase = "jedan dan" if days == 1 else f"{days} dana"
        second_user = (
            f"Za {destination_name} imam {day_phrase}, putujem {transport} tokom sezone {season}; "
            f"zadrži naglasak na {interest}."
        )
        excerpt = excerpt_from_chunks(selected, max_sentences=5, max_words=125, offset=index)
        final_answer = (
            f"Za {days} dana u destinaciji {destination_name} plan prilagodi interesovanju za {interest}. "
            f"{excerpt} Pošto putuješ {transport}, provjeri aktuelne uslove pristupa i prevoza prije polaska."
        )
        examples.append(
            make_example(
                example_id=f"SFT-MULTI-{index + 1:03d}",
                example_type="multi_turn",
                destination_ids=[destination_id],
                chunks=selected,
                question_family=f"multi_turn:{destination_id}:{interest}:{days}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": first_user},
                    {"role": "assistant", "content": "Koliko dana imaš i kojim prevozom planiraš putovati?"},
                    {"role": "user", "content": second_user},
                    {"role": "assistant", "content": final_answer},
                ],
            )
        )
    return examples


def generate_dynamic_refusals(
    destinations: Sequence[dict[str, str]],
    chunks_by_destination: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    count = int(config["additional_counts"]["uncertain_dynamic"])
    system_prompt = config["system_prompt"]
    examples: list[dict[str, Any]] = []

    for index in range(count):
        destination = destinations[index % len(destinations)]
        destination_id = destination["destination_id"]
        destination_name = destination["name"]
        selected = choose_chunks(
            chunks_by_destination[destination_id],
            ["practical", "route"],
            count=1,
        )
        chunk = selected[0]
        topic, object_phrase = DYNAMIC_TOPICS[index % len(DYNAMIC_TOPICS)]
        user = f"Možeš li mi navesti {object_phrase} za destinaciju „{destination_name}”?"
        answer = (
            f"Nemam dovoljno pouzdan, trenutno važeći podatak o temi „{topic}” za destinaciju "
            f"{destination_name}. Korišteni turistički kontekst posljednji put je provjeren "
            f"{chunk['last_verified_at']}, ali to nije potvrda današnjeg stanja. "
            "Prije putovanja provjeri službeni izvor; ne bih navodio broj, termin ili dostupnost bez nove potvrde."
        )
        examples.append(
            make_example(
                example_id=f"SFT-DYNAMIC-{index + 1:03d}",
                example_type="uncertain_dynamic",
                destination_ids=[destination_id],
                chunks=selected,
                question_family=f"dynamic:{topic}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
            )
        )
    return examples


def generate_out_of_domain(config: dict[str, Any]) -> list[dict[str, Any]]:
    count = int(config["additional_counts"]["out_of_domain"])
    system_prompt = config["system_prompt"]
    if count > len(OFF_DOMAIN_PROMPTS):
        raise ValueError(
            f"Requested {count} out-of-domain examples, but only "
            f"{len(OFF_DOMAIN_PROMPTS)} unique prompts are defined."
        )
    examples: list[dict[str, Any]] = []

    for index, user in enumerate(OFF_DOMAIN_PROMPTS[:count], start=1):
        answer = (
            "Specijalizovan sam za turističke informacije i planiranje putovanja u Bosni i Hercegovini. "
            "Za ovo pitanje nisam odgovarajući vodič, ali mogu pomoći oko destinacija, ruta, prevoza, "
            "aktivnosti i provjere turističkih izvora u BiH."
        )
        examples.append(
            make_example(
                example_id=f"SFT-OOD-{index:03d}",
                example_type="out_of_domain",
                destination_ids=[],
                chunks=[],
                question_family=f"out_of_domain:{index:03d}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
            )
        )
    return examples


def group_chunks_by_destination(
    chunks: Sequence[dict[str, Any]],
    destinations: Sequence[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    required = {
        "chunk_id",
        "destination_id",
        "destination_name",
        "categories",
        "text",
        "source_ids",
        "last_verified_at",
    }
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    known_destinations = {row["destination_id"] for row in destinations}
    seen_chunk_ids: set[str] = set()

    for position, chunk in enumerate(chunks, start=1):
        missing = required - set(chunk)
        if missing:
            raise ValueError(
                f"Knowledge-base chunk at position {position} is missing: "
                f"{', '.join(sorted(missing))}."
            )
        chunk_id = str(chunk["chunk_id"])
        destination_id = str(chunk["destination_id"])
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id {chunk_id}.")
        seen_chunk_ids.add(chunk_id)
        if destination_id not in known_destinations:
            raise ValueError(f"Unknown destination_id {destination_id} in chunk {chunk_id}.")
        if not chunk["source_ids"]:
            raise ValueError(f"Chunk {chunk_id} has no source_ids.")
        result[destination_id].append(chunk)

    missing_destinations = known_destinations - set(result)
    if missing_destinations:
        raise ValueError(
            "Destinations without knowledge-base chunks: "
            + ", ".join(sorted(missing_destinations))
        )
    return dict(result)


def all_user_messages(example: dict[str, Any]) -> list[str]:
    return [
        message["content"]
        for message in example["messages"]
        if message.get("role") == "user"
    ]


def validate_candidates(
    examples: Sequence[dict[str, Any]],
    destinations: Sequence[dict[str, str]],
    known_source_ids: set[str],
    minimum_total: int,
) -> list[str]:
    """Validate the generated T11 candidate collection."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_user_messages: set[str] = set()
    base_counts: Counter[str] = Counter()
    known_destination_ids = {row["destination_id"] for row in destinations}

    if len(examples) < minimum_total:
        errors.append(f"Expected at least {minimum_total} candidates, got {len(examples)}.")

    for position, example in enumerate(examples, start=1):
        missing = REQUIRED_FIELDS - set(example)
        if missing:
            errors.append(
                f"Example at position {position} is missing: {', '.join(sorted(missing))}."
            )
            continue

        example_id = example["example_id"]
        if example_id in seen_ids:
            errors.append(f"Duplicate example_id {example_id}.")
        seen_ids.add(example_id)

        if example["reviewed"] is not False or example["review_status"] != "pending":
            errors.append(f"{example_id} must start as reviewed=false and review_status=pending.")

        if not isinstance(example["messages"], list) or len(example["messages"]) < 3:
            errors.append(f"{example_id} must contain at least system, user and assistant messages.")
            continue

        roles = [message.get("role") for message in example["messages"]]
        if roles[0] != "system" or roles[-1] != "assistant":
            errors.append(f"{example_id} has invalid first or final role.")
        if example["type"] == "multi_turn" and roles != ROLE_SEQUENCE_MULTI_TURN:
            errors.append(f"{example_id} has invalid multi-turn role sequence.")

        for message in example["messages"]:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"{example_id} contains an empty message.")
            if isinstance(content, str) and ("{" in content or "}" in content):
                errors.append(f"{example_id} contains an unresolved template placeholder.")

        for user_message in all_user_messages(example):
            normalized_user = normalize_space(user_message).casefold()
            if normalized_user in seen_user_messages:
                errors.append(f"Duplicate user message in {example_id}: {user_message}")
            seen_user_messages.add(normalized_user)

        destination_ids = example["destination_ids"]
        if any(destination_id not in known_destination_ids for destination_id in destination_ids):
            errors.append(f"{example_id} references an unknown destination_id.")

        source_ids = example["source_ids"]
        if any(source_id not in known_source_ids for source_id in source_ids):
            errors.append(f"{example_id} references an unknown source_id.")
        if example["type"] not in NON_FACTUAL_TYPES and not source_ids:
            errors.append(f"{example_id} is factual but has no source_ids.")

        if example["type"] in DESTINATION_TYPES and len(destination_ids) == 1:
            base_counts[destination_ids[0]] += 1

    for destination_id in known_destination_ids:
        if base_counts[destination_id] < 8:
            errors.append(
                f"{destination_id} has {base_counts[destination_id]} destination candidates; required at least 8."
            )

    return errors


def build_stats(
    examples: Sequence[dict[str, Any]],
    destinations: Sequence[dict[str, str]],
    errors: Sequence[str],
) -> dict[str, Any]:
    type_counts = Counter(example["type"] for example in examples)
    destination_counts = Counter(
        destination_id
        for example in examples
        for destination_id in example["destination_ids"]
    )
    base_destination_counts = Counter(
        example["destination_ids"][0]
        for example in examples
        if example["type"] in DESTINATION_TYPES and len(example["destination_ids"]) == 1
    )
    user_messages = [
        normalize_space(message["content"]).casefold()
        for example in examples
        for message in example["messages"]
        if message["role"] == "user"
    ]
    return {
        "success": not errors,
        "candidate_count": len(examples),
        "type_counts": dict(sorted(type_counts.items())),
        "destination_candidate_counts": {
            destination["destination_id"]: destination_counts[destination["destination_id"]]
            for destination in destinations
        },
        "base_templates_per_destination": {
            destination["destination_id"]: base_destination_counts[destination["destination_id"]]
            for destination in destinations
        },
        "unique_user_message_count": len(set(user_messages)),
        "user_message_count": len(user_messages),
        "factual_candidate_count": sum(
            example["type"] not in NON_FACTUAL_TYPES for example in examples
        ),
        "critical_errors": list(errors),
    }


def generate_all_candidates(
    chunks: Sequence[dict[str, Any]],
    destinations: Sequence[dict[str, str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    chunks_by_destination = group_chunks_by_destination(chunks, destinations)
    examples: list[dict[str, Any]] = []
    examples.extend(generate_destination_examples(destinations, chunks_by_destination, config))
    examples.extend(generate_multi_destination_plans(destinations, chunks_by_destination, config))
    examples.extend(generate_clarifications(destinations, config))
    examples.extend(generate_multi_turn(destinations, chunks_by_destination, config))
    examples.extend(generate_dynamic_refusals(destinations, chunks_by_destination, config))
    examples.extend(generate_out_of_domain(config))
    return examples


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base", type=Path, default=KNOWLEDGE_BASE_PATH)
    parser.add_argument("--destinations", type=Path, default=DESTINATIONS_PATH)
    parser.add_argument("--templates", type=Path, default=TEMPLATES_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--stats-out", type=Path, default=STATS_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunks = read_jsonl(args.knowledge_base)
    destinations = read_destinations(args.destinations)
    config = load_templates(args.templates)
    examples = generate_all_candidates(chunks, destinations, config)

    known_source_ids = {
        source_id
        for chunk in chunks
        for source_id in chunk["source_ids"]
    }
    errors = validate_candidates(
        examples,
        destinations,
        known_source_ids,
        minimum_total=int(config["minimum_total_candidates"]),
    )
    stats = build_stats(examples, destinations, errors)
    write_json(args.stats_out, stats)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    write_jsonl(args.out, examples)
    print(
        f"SFT candidates generated successfully: {len(examples)} examples "
        f"for {len(destinations)} destinations."
    )
    print(f"Output: {args.out}")
    print(f"Stats: {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
