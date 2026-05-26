from __future__ import annotations

from mlip_workflows.cli import _looks_like_help, _print_help


def main(argv: list[str] | None = None) -> int:
    if _looks_like_help(argv):
        _print_help("mlip-snb")
        return 0
    from defect_landscape.snb.cli import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
