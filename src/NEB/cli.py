from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from .run_neb_raw_v2 import main as neb_main

    return neb_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
