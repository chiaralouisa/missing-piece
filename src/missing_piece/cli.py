"""Command-line entry point: ``python -m missing_piece <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment


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
        key, _, value = override.partition("=")
        if not hasattr(cfg, key):
            raise KeyError(f"unknown config field '{key}'")
        current = getattr(cfg, key)
        setattr(cfg, key, json.loads(value) if isinstance(current, (dict, list, int, float, bool)) or value.startswith(("{", "[")) else value)
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
