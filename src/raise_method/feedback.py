"""Deterministic formatting of verifier feedback for RAISE rollouts."""
from __future__ import annotations

from collections import defaultdict
import re

_UID = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)::"((?:[^"\\]|\\.)*)"')
_BLOCK = re.compile(r'([A-Za-z_][A-Za-z0-9_]*::"[^"]*")\s*\{([^{}]*)\}', re.DOTALL)
_SENTINELS = {"-1", '""', "false", "[]", "0", "{}"}


def enum_types_from_schema(schema_text: str) -> set[str]:
    """Return Cedar enum entity types whose instance IDs carry meaning."""
    return set(re.findall(r"entity\s+([A-Za-z_][A-Za-z0-9_]*)\s+enum\b", schema_text or ""))


def canonicalize_counterexample(raw: str, preserve_types: set[str] | None = None) -> str:
    """Anonymize opaque entity IDs while preserving actions and enum values."""
    if not raw:
        return ""
    keep = {"Action", *(preserve_types or set())}
    counters: dict[str, int] = defaultdict(int)
    names: dict[tuple[str, str], str] = {}

    def replace(match: re.Match[str]) -> str:
        entity_type, identifier = match.groups()
        if entity_type in keep:
            return match.group(0)
        key = (entity_type, identifier)
        if key not in names:
            abbreviation = "".join(c for c in entity_type if c.isupper()) or entity_type[:1].upper()
            names[key] = f'{entity_type}::"{abbreviation}{counters[entity_type]}"'
            counters[entity_type] += 1
        return names[key]

    return _UID.sub(replace, raw)


def attr_names_in(description: str) -> set[str]:
    """Find likely Cedar attribute names mentioned by a check description."""
    return set(re.findall(r"[a-z][a-zA-Z0-9]{2,}", description or ""))


def _is_sentinel(value: str) -> bool:
    value = value.strip().rstrip(",").strip()
    return value in _SENTINELS or "1970-01-01" in value


def _split_attributes(body: str) -> list[tuple[str, str]]:
    parts, depth, current = [], 0, ""
    for char in body:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current)
    return [tuple(item.split(":", 1)) for item in parts if ":" in item]


def reduce_counterexample(raw: str, keep_names: set[str] | None = None) -> str:
    """Keep request facts and non-default entity attributes from a solver model."""
    if not raw or not raw.strip():
        return ""
    keep = keep_names or set()
    principal = re.search(r"^principal:.*$", raw, re.MULTILINE)
    context = re.search(r"^context:.*$", raw, re.MULTILINE)
    entities_at = raw.find("entities:")
    if not principal or entities_at < 0:
        return raw.replace("\n", " ").strip()

    principal_line = principal.group(0).strip()
    context_line = context.group(0).strip() if context else ""
    blocks = {}
    order = []
    for uid, body in _BLOCK.findall(raw[entities_at:]):
        blocks[uid] = [(key.strip(), value.strip()) for key, value in _split_attributes(body)]
        order.append(uid)
    if not blocks:
        return (principal_line + (" " + context_line if context_line else "")).strip()

    roots = {f'{kind}::"{identifier}"' for kind, identifier in
             _UID.findall(principal_line + " " + context_line)}
    seen, frontier = set(), [uid for uid in roots if uid in blocks]
    while frontier:
        uid = frontier.pop()
        if uid in seen:
            continue
        seen.add(uid)
        for _, value in blocks[uid]:
            for kind, identifier in _UID.findall(value):
                child = f'{kind}::"{identifier}"'
                if child in blocks and child not in seen:
                    frontier.append(child)
    if not seen:
        seen = set(blocks)

    trimmed = {
        uid: [(key, value) for key, value in blocks[uid]
              if key in keep or not _is_sentinel(value)]
        for uid in order if uid in seen
    }
    groups: dict[str, list[str]] = defaultdict(list)
    for uid, attributes in trimmed.items():
        signature = "; ".join(f"{key}: {value}" for key, value in attributes)
        groups[signature].append(uid)
    entity_text = "; ".join(
        f"{', '.join(uids)} {{{signature}}}" if signature else ", ".join(uids)
        for signature, uids in groups.items()
    )
    result = principal_line
    if context_line:
        result += " " + context_line
    if entity_text:
        result += " entities: " + entity_text
    return re.sub(r"\s+", " ", result).strip()
