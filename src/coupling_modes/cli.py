from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from .phonon_coupling import main as coupling_main

    return coupling_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
