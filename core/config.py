"""Layered, strictly-validated configuration loading.

Implements the precedence chain described in the FitSpec architecture
docs (lowest to highest):

1. Base default config (``default_config.dat``) -- shared across all
   fitting modes.
2. Mode-specific default config (``default_config_{mode}.dat``) --
   overrides the base default on any shared key.
3. User config (``config.dat``) -- overrides any default for this run.
4. Command-line arguments -- override both, auto-generated from
   whatever keys exist in the default config files.

Levels 1+2 together are the *complete, authoritative* set of recognized
keywords: every keyword FitSpec will ever read must have a default
value declared in one of those four files. That default value's Python
type becomes the schema entry for that key, and every value from
level 3 or 4 is coerced against it. A keyword absent from every default
file is a fatal ``ConfigError`` at any layer -- never a warning, never
silently accepted.

File format is unchanged from the legacy scripts: plain-text
``key = value`` lines, ``#`` starts a comment (to end of line, inline or
whole-line), comma-separated values become lists. Unlike the legacy
parser, type inference only happens once, while establishing the schema
from the default files; every subsequent layer is coerced against that
established type rather than independently re-guessed, and there is no
hardcoded per-keyword special case (the legacy parser hardcoded
``['strong_lines', 'weak_lines']`` as the only list-valued keys).
"""
from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


__all__ = ["ConfigError", "SchemaEntry", "Config", "load_config"]


class ConfigError(Exception):
    """Raised for any unknown keyword or type-coercion failure, at any layer."""


@dataclass
class SchemaEntry:
    """The recognized type for one config key, established from its example/default."""

    value_type: type
    element_type: "type | None" = None  # set when value_type is list
    nested_element_type: "type | None" = None  # list-of-lists, e.g. intervals


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def _infer_scalar(token: str):
    token = token.strip()
    lowered = token.lower()
    # Preserve zero-padded identifiers such as BPASS metallicity codes
    # (0001, 0008, 0014) as strings rather than silently converting them
    # to integers and losing their library-key semantics.
    if len(token) > 1 and token.isdigit() and token.startswith("0"):
        return token
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _infer_value(token: str):
    token = token.strip()
    # Bracketed literals are useful for interval lists such as
    # [[1140,1147],[1180,1224]].  Keep the ordinary legacy comma syntax for
    # simple one-dimensional lists.
    if token.startswith("[") and token.endswith("]"):
        try:
            value = ast.literal_eval(token)
        except (ValueError, SyntaxError):
            value = None
        if isinstance(value, (list, tuple)):
            return list(value)
    if "," in token:
        return [_infer_scalar(part) for part in token.split(",")]
    return _infer_scalar(token)


def _coerce_scalar(token: str, value_type: type, *, context: str):
    token = token.strip()
    if value_type is bool:
        lowered = token.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ConfigError(f"{context}: expected a boolean, got {token!r}.")
    if value_type is int:
        try:
            return int(token)
        except ValueError:
            raise ConfigError(f"{context}: expected an integer, got {token!r}.")
    if value_type is float:
        try:
            return float(token)
        except ValueError:
            raise ConfigError(f"{context}: expected a float, got {token!r}.")
    return token  # str


def _coerce_value(token: str, schema_entry: SchemaEntry, *, context: str):
    token = token.strip()
    if schema_entry.value_type is list:
        if schema_entry.element_type is list:
            try:
                value = ast.literal_eval(token)
            except (ValueError, SyntaxError) as exc:
                raise ConfigError(f"{context}: expected a bracketed list of lists, got {token!r}.") from exc
            if not isinstance(value, (list, tuple)):
                raise ConfigError(f"{context}: expected a list of lists, got {token!r}.")
            result = []
            for row in value:
                if not isinstance(row, (list, tuple)):
                    raise ConfigError(f"{context}: every nested-list entry must itself be a list/tuple.")
                result.append([
                    _coerce_scalar(str(item), schema_entry.nested_element_type or float, context=context)
                    for item in row
                ])
            return result
        parts = [p.strip() for p in token.split(",")] if token else []
        return [_coerce_scalar(p, schema_entry.element_type, context=context) for p in parts]
    return _coerce_scalar(token, schema_entry.value_type, context=context)


def _build_schema(defaults: dict) -> "dict[str, SchemaEntry]":
    schema = {}
    for key, value in defaults.items():
        if isinstance(value, list):
            element_type = type(value[0]) if value else str
            nested_element_type = None
            if element_type is list and value and value[0]:
                nested_element_type = type(value[0][0])
            schema[key] = SchemaEntry(
                value_type=list, element_type=element_type,
                nested_element_type=nested_element_type,
            )
        else:
            schema[key] = SchemaEntry(value_type=type(value))
    return schema


def _parse_default_file(path: Path) -> dict:
    """Parse a default config file, inferring types (establishes the schema)."""
    if not path.is_file():
        raise ConfigError(f"Required default config file not found: {path}")
    values = {}
    with open(path, "r") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = _strip_comment(raw_line).strip()
            if not line:
                continue
            if "=" not in line:
                raise ConfigError(f"{path}:{line_number}: expected 'key = value', got {raw_line!r}.")
            key, _, value = line.partition("=")
            key = key.strip()
            values[key] = _infer_value(value)
    return values


_OPTIONAL_SCHEMA_RE = re.compile(
    r"^\s*#\s*OPTIONAL\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$",
    flags=re.IGNORECASE,
)


def _parse_optional_schema_file(path: Path) -> dict:
    """Return commented ``# OPTIONAL: key = example`` schema declarations.

    Optional declarations are deliberately *not* inserted into the runtime
    configuration unless the user supplies the key in ``config.dat`` or on
    the command line.  Their example value exists only to establish the strict
    type schema and to make every supported keyword discoverable in the
    shipped default config files.
    """
    values = {}
    with open(path, "r") as handle:
        for raw_line in handle:
            match = _OPTIONAL_SCHEMA_RE.match(raw_line.rstrip("\n"))
            if not match:
                continue
            key, token = match.groups()
            values[key] = _infer_value(token)
    return values


def _schema_defaults_for_files(*paths: Path) -> dict:
    """Merge active defaults plus commented optional schema examples."""
    schema_defaults = {}
    for path in paths:
        active = _parse_default_file(path)
        optional = _parse_optional_schema_file(path)
        # Active declarations are authoritative when a key is documented in
        # both forms (e.g. an example repeated in nearby comments).
        schema_defaults.update(optional)
        schema_defaults.update(active)
    return schema_defaults


def _parse_overlay_file(path: Path, schema: "dict[str, SchemaEntry]") -> dict:
    """Parse a config.dat-style overlay, strictly validated against schema."""
    values = {}
    with open(path, "r") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = _strip_comment(raw_line).strip()
            if not line:
                continue
            if "=" not in line:
                raise ConfigError(f"{path}:{line_number}: expected 'key = value', got {raw_line!r}.")
            key, _, value = line.partition("=")
            key = key.strip()
            context = f"{path}:{line_number} (key {key!r})"
            if key not in schema:
                raise ConfigError(
                    f"{context}: unrecognized config keyword. Every keyword must have a "
                    "default declared in a default_config*.dat file."
                )
            values[key] = _coerce_value(value, schema[key], context=context)
    return values


class _StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ConfigError(f"command line: {message}")


def _build_arg_parser(schema: "dict[str, SchemaEntry]") -> argparse.ArgumentParser:
    parser = _StrictArgumentParser(add_help=True)
    for key, entry in schema.items():
        flag = f"--{key}"
        if entry.value_type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        else:
            # Always accept as a raw string here and coerce with the same
            # _coerce_value used for config.dat, so CLI and file overlays
            # are parsed identically rather than via two separate paths.
            parser.add_argument(flag, type=str, default=argparse.SUPPRESS)
    return parser


@dataclass
class Config:
    """Final, merged configuration, with provenance tracked per key."""

    values: dict
    sources: dict  # key -> one of "default_base", "default_mode", "user_config", "cli"

    def __getitem__(self, key):
        return self.values[key]

    def __contains__(self, key):
        return key in self.values

    def __iter__(self):
        return iter(self.values)

    def items(self):
        return self.values.items()

    def get(self, key, default=None):
        return self.values.get(key, default)

    def source_of(self, key) -> str:
        return self.sources[key]


def load_config(
    mode: str, config_dir, *, run_dir=None,
    base_default_filename: str = "default_config.dat",
    mode_default_filename_template: str = "default_config_{mode}.dat",
    user_config_filename: str = "config.dat",
    cli_args: "list[str] | None" = None,
) -> Config:
    """Load configuration for one fitting mode with the full precedence chain.

    Parameters
    ----------
    mode : str
        One of the fitting modes (e.g. "stellar", "emission", "absorption");
        selects ``default_config_{mode}.dat``.
    config_dir : str or Path
        Directory containing the default config files.
    run_dir : str or Path, optional
        Directory to look for an optional user ``config.dat`` in. If not
        given, or the file doesn't exist there, only the defaults (and
        any CLI overrides) apply.
    cli_args : list[str], optional
        Command-line arguments to parse (e.g. ``sys.argv[1:]``); if None,
        no CLI layer is applied at all (useful for testing / non-CLI use).

    Returns
    -------
    Config
    """
    config_dir = Path(config_dir)

    base_defaults = _parse_default_file(config_dir / base_default_filename)
    mode_defaults = _parse_default_file(config_dir / mode_default_filename_template.format(mode=mode))

    values = dict(base_defaults)
    sources = {key: "default_base" for key in base_defaults}
    for key, value in mode_defaults.items():
        values[key] = value
        sources[key] = "default_mode"

    schema_defaults = _schema_defaults_for_files(
        config_dir / base_default_filename,
        config_dir / mode_default_filename_template.format(mode=mode),
    )
    schema = _build_schema(schema_defaults)

    if run_dir is not None:
        user_path = Path(run_dir) / user_config_filename
        if user_path.is_file():
            user_values = _parse_overlay_file(user_path, schema)
            for key, value in user_values.items():
                values[key] = value
                sources[key] = "user_config"

    if cli_args is not None:
        parser = _build_arg_parser(schema)
        namespace = parser.parse_args(cli_args)
        for key, raw_value in vars(namespace).items():
            entry = schema[key]
            if entry.value_type is bool:
                value = raw_value  # BooleanOptionalAction already yields a bool
            else:
                value = _coerce_value(str(raw_value), entry, context=f"command line (--{key})")
            values[key] = value
            sources[key] = "cli"

    return Config(values=values, sources=sources)


def load_configs(
    modes, config_dir, *, run_dir=None,
    base_default_filename: str = "default_config.dat",
    mode_default_filename_template: str = "default_config_{mode}.dat",
    user_config_filename: str = "config.dat",
) -> "dict[str, Config]":
    """Load several mode configs from one strictly validated user config.

    ``load_config`` intentionally validates one mode at a time.  A unified
    FitSpec application, however, needs one ``config.dat`` that may contain
    stellar, emission, and absorption keys together.  This function builds the
    union schema across the requested modes, validates the user overlay once,
    and then applies each key only to mode configurations that recognize it.
    Unknown keys remain fatal and shared-key type conflicts are rejected.
    """
    modes = tuple(modes)
    config_dir = Path(config_dir)
    base_defaults = _parse_default_file(config_dir / base_default_filename)
    mode_defaults = {
        mode: _parse_default_file(config_dir / mode_default_filename_template.format(mode=mode))
        for mode in modes
    }

    per_mode_defaults = {}
    per_mode_schema = {}
    base_path = config_dir / base_default_filename
    union_defaults = dict(base_defaults)
    union_schema = _build_schema(_schema_defaults_for_files(base_path))
    for mode in modes:
        defaults = dict(base_defaults)
        defaults.update(mode_defaults[mode])
        per_mode_defaults[mode] = defaults
        mode_path = config_dir / mode_default_filename_template.format(mode=mode)
        schema = _build_schema(_schema_defaults_for_files(base_path, mode_path))
        per_mode_schema[mode] = schema
        for key, entry in schema.items():
            if key in union_schema:
                old = union_schema[key]
                if (old.value_type, old.element_type, old.nested_element_type) != (
                    entry.value_type, entry.element_type, entry.nested_element_type
                ):
                    raise ConfigError(
                        f"Config key {key!r} has incompatible declared types across FitSpec modes."
                    )
            else:
                union_schema[key] = entry

    overlay = {}
    if run_dir is not None:
        user_path = Path(run_dir) / user_config_filename
        if user_path.is_file():
            overlay = _parse_overlay_file(user_path, union_schema)

    configs = {}
    for mode in modes:
        values = dict(per_mode_defaults[mode])
        sources = {key: ("default_mode" if key in mode_defaults[mode] else "default_base") for key in values}
        for key, value in overlay.items():
            if key in per_mode_schema[mode]:
                values[key] = value
                sources[key] = "user_config"
        configs[mode] = Config(values=values, sources=sources)
    return configs


__all__.append("load_configs")
