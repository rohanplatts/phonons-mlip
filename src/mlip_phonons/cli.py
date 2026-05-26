from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from .main import main as phonons_main

    return phonons_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
