from __future__ import annotations

import argparse
import sys


def _looks_like_help(argv: list[str] | None) -> bool:
    args = sys.argv[1:] if argv is None else list(argv)
    return any(token in {"-h", "--help"} for token in args)


def _print_help(prog: str) -> None:
    print(f"usage: {prog} [--inputs INPUTS] [--outputs OUTPUTS]")
    print()
    print("options:")
    print("  --inputs INPUTS   Path to a config directory.")
    print("  --outputs OUTPUTS  Optional output directory.")


def parse_common(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--inputs")
    return parser.parse_known_args(argv)
