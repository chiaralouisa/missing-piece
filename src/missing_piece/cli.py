"""Command-line entry point: ``python -m missing_piece <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment


def _coerce(value: str, current):
    """Parse a --set value to match the type of the field it overrides.

    Strings stay strings (so paths and names need no quoting); anything else is
    read as JSON, which covers ints, floats, booleans, null, lists and the
    nested dicts used for model and split settings.
    """
    if isinstance(current, str) or current is None and not value.startswith(("{", "[")):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse {value!r} as JSON: {exc}") from None


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = (
        ExperimentConfig.from_yaml(args.config)
        if args.config
        else ExperimentConfig()
    )
    for override in args.set or []:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {override!r}")
        if key not in ExperimentConfig.__dataclass_fields__:
            raise KeyError(
                f"unknown config field {key!r}; "
                f"available: {sorted(ExperimentConfig.__dataclass_fields__)}"
            )
        setattr(cfg, key, _coerce(value, getattr(cfg, key)))
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.name:
        cfg.name = args.name

    result = run_experiment(cfg)
    print(result.render())
    out = result.save()
    print(f"\nwritten to {out}", file=sys.stderr)
    return 0


def _cmd_panels(args: argparse.Namespace) -> int:
    from .panels import PanelPair

    pair = PanelPair.from_directory(args.panel_dir, args.observed, args.target)
    print(pair.describe())
    for warning in pair.validate():
        print(f"WARNING: {warning}", file=sys.stderr)
    if args.list_target_genes:
        print("\ntarget genes:")
        for gene in pair.target_genes:
            print(f"  {gene}")
    return 0


def _cmd_controls(args: argparse.Namespace) -> int:
    """Run the negative-control battery."""
    from .validation import run_negative_controls

    report = run_negative_controls(
        n_patients=args.n_patients,
        seed=args.seed,
        models=args.models,
        output_dir=args.output_dir,
    )
    print(report)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .inspect_study import inspect_study

    report = inspect_study(
        args.study_dir,
        panel_dir=args.panel_dir,
        observed_panel=args.observed,
        target_panel=args.target,
        restrict_to_nsclc=not args.all_cancer_types,
    )
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"\nwritten to {args.out}", file=sys.stderr)
    return 0 if "FATAL" not in report else 1


def _cmd_export(args: argparse.Namespace) -> int:
    from .export import export_cohort

    path = export_cohort(
        args.study_dir,
        out_path=args.out,
        panel_dir=args.panel_dir,
        observed_panel=args.observed,
        target_panel=args.target,
        restrict_to_nsclc=not args.all_cancer_types,
    )
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="missing_piece", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a panel-completion experiment")
    run.add_argument("-c", "--config", type=Path, help="YAML experiment config")
    run.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config field")
    run.add_argument("--output-dir")
    run.add_argument("--name")
    run.set_defaults(func=_cmd_run)

    panels = sub.add_parser("panels", help="describe a panel-completion task")
    panels.add_argument("panel_dir", type=Path)
    panels.add_argument("--observed", default="IMPACT341")
    panels.add_argument("--target", default="IMPACT505")
    panels.add_argument("--list-target-genes", action="store_true")
    panels.set_defaults(func=_cmd_panels)

    inspect = sub.add_parser(
        "inspect", help="check a study directory loads correctly (run this first)"
    )
    inspect.add_argument("study_dir", type=Path)
    inspect.add_argument("--panel-dir", type=Path)
    inspect.add_argument("--observed", default="IMPACT341")
    inspect.add_argument("--target", default="IMPACT505")
    inspect.add_argument("--all-cancer-types", action="store_true")
    inspect.add_argument("--out", help="also write the report to this file")
    inspect.set_defaults(func=_cmd_inspect)

    export = sub.add_parser(
        "export", help="write the derived cohort as a single portable file"
    )
    export.add_argument("study_dir", type=Path)
    export.add_argument("-o", "--out", default="cohort_nsclc.npz")
    export.add_argument("--panel-dir", type=Path)
    export.add_argument("--observed", default="IMPACT341")
    export.add_argument("--target", default="IMPACT505")
    export.add_argument("--all-cancer-types", action="store_true")
    export.set_defaults(func=_cmd_export)

    controls = sub.add_parser("controls", help="run the negative-control battery")
    controls.add_argument("--n-patients", type=int, default=2226)
    controls.add_argument("--seed", type=int, default=0)
    controls.add_argument("--models", nargs="*", default=["prevalence", "burden", "logistic", "flow_discrete"])
    controls.add_argument("--output-dir", default="results")
    controls.set_defaults(func=_cmd_controls)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
