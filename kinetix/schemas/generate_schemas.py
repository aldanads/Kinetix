#!/usr/bin/env python3
"""Generate JSON Schema (Draft 2020-12) files from the Kinetix config dataclasses.

Why this exists
---------------
The YAML files under ``data/parameters/`` are the de-facto data contract of the
multi-scale pipeline, but that contract is only expressed as Python dataclasses.
This script reads those dataclasses and writes a machine-readable JSON Schema per
class, so external tools (editors, validators, workflow managers) 
can check a configuration without importing Python.

Design constraints
------------------
* **Read-only**: nothing in ``kinetix/`` is modified - the dataclasses stay the
  single source of truth and the schemas are generated artefacts.
* **Standard library only.** ``pydantic`` could do this, but it is not a
  declared dependency of Kinetix and a dev tool must not add one.
* **Faithful, not strict.** The production loaders ignore unknown YAML keys
  (``from_dict`` reads the keys it knows), so the schemas deliberately do not
  forbid extra properties: they document the known fields without breaking real
  preset files that carry additional sections.

Usage
-----
    python kinetix/schemas/generate_schemas.py            # write the schemas
    python kinetix/schemas/generate_schemas.py --check    # write nothing, just validate
    python kinetix/schemas/generate_schemas.py --validate-presets

The optional ``--validate-presets`` step needs the ``jsonschema`` package; the
schema *generation* itself never does.
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import inspect
import json
import pathlib
import re
import sys
import types
import typing
from typing import Any

# --- bootstrap ---------------------------------------------------------------
# Allow `python kinetix/schemas/generate_schemas.py` from any directory: running
# a script puts *its* directory on sys.path, not the repository root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import kinetix.configs as configs_pkg  # noqa: E402

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_BASE_URI = "https://github.com/aldanads/Kinetix/schemas"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent
CONFIG_PACKAGE = "kinetix.configs"

# Fields injected by the runtime rather than read from YAML. They are real
# dataclass fields but not part of the file contract, so the schemas mark them
# instead of hiding them.
RUNTIME_INJECTED = {"rng", "mpi_ctx", "base_path", "config_path", "yaml_path"}


# --- discovery ---------------------------------------------------------------
def discover_config_classes() -> list[type]:
    """Return every dataclass defined in ``kinetix/configs/``, sorted by name."""
    classes: list[type] = []
    for path in sorted(pathlib.Path(configs_pkg.__path__[0]).glob("*.py")):
        if path.stem == "__init__":
            continue
        module = importlib.import_module(f"{CONFIG_PACKAGE}.{path.stem}")
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (dataclasses.is_dataclass(obj)
                    and obj.__module__ == module.__name__):
                classes.append(obj)
    return sorted(set(classes), key=lambda cls: (cls.__module__, cls.__name__))


# --- docstring helpers -------------------------------------------------------
_ATTRIBUTE_HEADER = re.compile(r"^\s*(Attributes|Args|Parameters)\s*:\s*$")
_FIELD_DOC_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$")


def parse_attribute_docs(cls: type) -> dict[str, str]:
    """Parse a Google-style ``Attributes:`` block of a class docstring.

    Only ``Attributes:`` is interpreted: it documents *fields*, which is what a
    data contract needs (``Args:``/``Parameters:`` document methods).
    """
    doc = inspect.getdoc(cls) or ""
    lines = doc.splitlines()
    docs: dict[str, str] = {}
    for index, line in enumerate(lines):
        header = _ATTRIBUTE_HEADER.match(line)
        if not header or header.group(1) != "Attributes":
            continue
        base_indent = len(line) - len(line.lstrip()) + 4
        current: str | None = None
        for raw in lines[index + 1:]:
            if not raw.strip():
                continue
            indent = len(raw) - len(raw.lstrip())
            if indent < base_indent:
                break
            match = _FIELD_DOC_RE.match(raw)
            if match and indent == base_indent:
                current = match.group(1)
                docs[current] = match.group(2).strip()
            elif current:
                docs[current] = f"{docs[current]} {raw.strip()}".strip()
        break
    return docs


def parse_inline_comments(cls: type) -> dict[str, str]:
    """Trailing ``# ...`` comments on field lines (``temperature: float = 300.0  # K``)."""
    try:
        source = inspect.getsource(cls)
    except OSError:  # pragma: no cover - interactive use
        return {}
    comments: dict[str, str] = {}
    for line in source.splitlines():
        match = re.match(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s*:[^=]*=.*?#\s*(\S.*)$", line)
        if match:
            comments.setdefault(match.group(1), match.group(2).strip())
    return comments


class SchemaBuilder:
    """Translate dataclass field types into JSON Schema, collecting ``$defs``."""

    def __init__(self) -> None:
        self.defs: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _def_name(tp: Any) -> str:
        return getattr(tp, "__name__", str(tp))

    def _register(self, tp: type) -> None:
        """Emit a ``$defs`` entry for a nested dataclass or enum."""
        name = self._def_name(tp)
        if name in self.defs:
            return
        if isinstance(tp, type) and issubclass(tp, enum.Enum):
            # The configuration files select enum members by NAME
            # (``VoltageMode[mode_str]`` in electrical_config.from_dict), and
            # the members of VoltageMode/CurrentModel are ``auto()`` integers,
            # so the names - not the values - are the file contract. The
            # underlying values are kept in a non-standard annotation.
            entry: dict[str, Any] = {
                "title": name,
                "enum": [member.name for member in tp],
                "x-enum-values": {member.name: member.value for member in tp},
                "description": "Selected by member NAME in the YAML files.",
            }
            doc = (inspect.getdoc(tp) or "").strip()
            if doc:
                entry["description"] = f"{doc} Selected by member NAME in the YAML files."
            self.defs[name] = entry
            return
        if dataclasses.is_dataclass(tp):
            # Reserve the slot first, so a self-referential config cannot recurse.
            self.defs[name] = {"title": name}
            self.defs[name] = self.object_schema(tp)
            return

    def _ref(self, tp: type) -> dict[str, Any]:
        self._register(tp)
        return {"$ref": f"#/$defs/{self._def_name(tp)}"}

    def type_schema(self, tp: Any) -> dict[str, Any]:
        """Map a single (non-nullable) type to a JSON Schema fragment."""
        if tp is Any or tp is None or tp is type(None):
            return {}
        if tp is str:
            return {"type": "string"}
        if tp is bool:
            return {"type": "boolean"}
        if tp is int:
            return {"type": "integer"}
        if tp is float:
            return {"type": "number"}
        if isinstance(tp, type) and issubclass(tp, pathlib.PurePath):
            return {"type": "string", "description": "Filesystem path (YAML: a string)."}
        origin = typing.get_origin(tp)
        if origin in (list, set, frozenset):
            args = typing.get_args(tp)
            return {"type": "array", "items": self.type_schema(args[0]) if args else {}}
        if origin is tuple:
            args = typing.get_args(tp)
            if not args:
                return {"type": "array"}
            if len(args) == 2 and args[1] is Ellipsis:        # tuple[X, ...]
                return {"type": "array", "items": self.type_schema(args[0])}
            return {"type": "array",                         # fixed-length tuple
                    "prefixItems": [self.type_schema(a) for a in args],
                    "minItems": len(args), "maxItems": len(args)}
        if origin is dict:
            args = typing.get_args(tp)
            return {"type": "object",
                    "additionalProperties": self.type_schema(args[1]) if len(args) == 2 else {}}
        if isinstance(tp, type) and issubclass(tp, enum.Enum):
            return self._ref(tp)
        if dataclasses.is_dataclass(tp):
            return self._ref(tp)
        return {}                                               # unannotated / unknown

    def field_schema(self, tp: Any) -> dict[str, Any]:
        """Map a field type, splitting ``X | None`` into an ``anyOf`` with null."""
        origin = typing.get_origin(tp)
        if origin in (types.UnionType, typing.Union):
            all_args = typing.get_args(tp)
            args = [a for a in all_args if a is not type(None)]
            nullable = len(args) != len(all_args)
            if len(args) == 1:
                schema = self.type_schema(args[0])
            else:
                schema = {"anyOf": [self.type_schema(a) for a in args]}
            return {"anyOf": [schema, {"type": "null"}]} if nullable else schema
        if origin is typing.Literal:                            # Literal['a', 'b']
            return {"enum": list(typing.get_args(tp))}
        return self.type_schema(tp)

    def object_schema(self, cls: type) -> dict[str, Any]:
        """Build the object schema of one dataclass (recursing into ``$defs``)."""
        hints = typing.get_type_hints(cls)
        attr_docs = parse_attribute_docs(cls)
        inline_docs = parse_inline_comments(cls)

        properties: dict[str, Any] = {}
        required: list[str] = []
        for field in dataclasses.fields(cls):
            schema = self.field_schema(hints.get(field.name, field.type))
            description = attr_docs.get(field.name) or inline_docs.get(field.name)
            if description:
                schema = {**schema, "description": description.rstrip()}
            if field.name in RUNTIME_INJECTED:
                note = "Injected by the runtime, not read from the YAML file."
                text = f"{description} {note}".strip() if description else note
                schema = {**schema, "x-runtime-injected": True, "description": text}
            if field.default is not dataclasses.MISSING:
                if _json_safe(field.default):
                    schema = {**schema, "default": field.default}
            elif field.default_factory is not dataclasses.MISSING:   # type: ignore[misc]
                try:
                    produced = field.default_factory()                # type: ignore[misc]
                except Exception:  # noqa: BLE001 - a factory may need context
                    produced = None
                if _json_safe(produced):
                    schema = {**schema, "default": produced}
            else:
                required.append(field.name)
            properties[field.name] = schema

        out: dict[str, Any] = {
            "type": "object",
            "title": cls.__name__,
            "properties": properties,
            "additionalProperties": False,
        }
        doc = (inspect.getdoc(cls) or "").strip()
        if doc:
            out["description"] = doc
        if required:
            out["required"] = required
        return out


def _json_safe(value: Any) -> bool:
    """True when ``value`` can be embedded in a JSON document."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def build_schema(cls: type) -> dict[str, Any]:
    """Full Draft 2020-12 document for one dataclass."""
    builder = SchemaBuilder()
    root = builder.object_schema(cls)
    name = snake_case(cls.__name__)
    document = {"$schema": SCHEMA_DIALECT,
                "$id": f"{SCHEMA_BASE_URI}/{name}.schema.json", **root}
    if builder.defs:
        document["$defs"] = builder.defs
    return document


# --- validation (optional, needs the ``jsonschema`` package) ------------------
def validate_documents(documents: dict[str, dict[str, Any]]) -> list[str]:
    """Meta-schema-validate every document. Returns a list of problem strings."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return ["jsonschema not installed - meta-schema validation skipped"]
    problems: list[str] = []
    for filename, document in documents.items():
        try:
            Draft202012Validator.check_schema(document)
        except Exception as exc:  # noqa: BLE001 - report any schema failure
            problems.append(f"{filename}: {type(exc).__name__}: {exc}")
    return problems


# Sections of a preset file that map 1:1 onto a typed config object. The other
# top-level keys (``metadata``, ``crystal``, ``components``) form the YAML
# *envelope*: ``SimulationConfig.from_yaml`` folds ``metadata`` into
# name/description/author, merges ``crystal`` into the material config, and uses
# ``components`` to load the per-family YAML files (defects, reactions,
# grain_boundaries, electrical) - each of which has its own schema here.
PRESET_SECTION_SCHEMAS = {
    "settings": "simulation_settings.schema.json",
    "experimental": "experimental_conditions.schema.json",
    "superbasin": "superbasin_config.schema.json",
    "poisson": "poisson_solver_config.schema.json",
    "heat": "heat_solver_config.schema.json",
    "mesh": "mesh_config.schema.json",
    "calculator": "calculator_config.schema.json",
}
PRESET_ENVELOPE_KEYS = ("metadata", "crystal", "components")

# Component files, validated with the schema of the config they feed.
COMPONENT_SCHEMAS = {
    "defects": "defects_config.schema.json",
    "reactions": "reactions_config.schema.json",
    "grain_boundaries": "grain_boundaries_config.schema.json",
    "electrical": "electrical_config.schema.json",
}


def _first_error(validator_cls, data, schema_name: str, documents) -> str | None:
    """First validation error of ``data`` against a named schema, or None."""
    schema = documents.get(schema_name)
    if schema is None:
        return f"{schema_name} missing"
    errors = sorted(validator_cls(schema).iter_errors(data), key=lambda e: list(e.path))
    if not errors:
        return None
    first = errors[0]
    location = "/".join(str(p) for p in first.path) or "<root>"
    return f"{schema_name} at {location}: {first.message}"


def validate_presets(documents: dict[str, dict[str, Any]]) -> list[str]:
    """Validate the shipped YAML files against the generated schemas.

    Presets are checked section by section (the keys that map 1:1 onto a typed
    config) and component files against the schema of the config they feed. The
    YAML envelope keys are reported rather than validated: no dataclass
    describes them, because ``SimulationConfig.from_yaml`` consumes them.
    """
    try:
        import yaml
        from jsonschema import Draft202012Validator
    except ImportError:
        return ["jsonschema not installed - preset validation skipped"]

    results: list[str] = []
    params = _REPO_ROOT / "data" / "parameters"
    for preset in sorted((params / "presets").glob("*.yaml")):
        data = yaml.safe_load(preset.read_text()) or {}
        problems: list[str] = []
        checked = 0
        for section, schema_name in PRESET_SECTION_SCHEMAS.items():
            if section not in data:
                continue
            checked += 1
            problem = _first_error(Draft202012Validator, data[section], schema_name, documents)
            if problem:
                problems.append(problem)
        envelope = ", ".join(k for k in PRESET_ENVELOPE_KEYS if k in data) or "none"
        if problems:
            status = f"{len(problems)} problem(s); first: {problems[0]}"
        else:
            status = f"{checked} typed section(s) valid"
        results.append(f"{preset.name}: {status} (envelope: {envelope})")

    for folder, schema_name in COMPONENT_SCHEMAS.items():
        for path in sorted((params / folder).glob("*.yaml")):
            data = yaml.safe_load(path.read_text()) or {}
            problem = _first_error(Draft202012Validator, data, schema_name, documents)
            rel = path.relative_to(params)
            results.append(f"{rel}: valid" if problem is None else f"{rel}: {problem}")
    return results


# --- entry point -------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate JSON Schemas from the config dataclasses")
    parser.add_argument("--check", action="store_true",
                        help="do not write files; only validate and report")
    parser.add_argument("--validate-presets", action="store_true",
                        help="also validate data/parameters/presets/*.yaml")
    args = parser.parse_args(argv)

    documents = {f"{snake_case(cls.__name__)}.schema.json": build_schema(cls)
                 for cls in discover_config_classes()}

    if not args.check:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for filename, document in sorted(documents.items()):
            (OUTPUT_DIR / filename).write_text(
                json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    problems = validate_documents(documents)
    print(f"{'checked' if args.check else 'wrote'} {len(documents)} schemas in {OUTPUT_DIR}")
    for filename in sorted(documents):
        document = documents[filename]
        print(f"  {filename:36s} {len(document.get('properties', {})):3d} properties"
              f"  {len(document.get('$defs', {})):2d} $defs")
    if problems:
        print("\nVALIDATION PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nAll schemas are valid Draft 2020-12 documents.")
    if args.validate_presets:
        print("\nPreset validation against simulation_config.schema.json:")
        for line in validate_presets(documents):
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
