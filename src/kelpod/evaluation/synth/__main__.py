"""``python -m kelpod.evaluation.synth --out samples --n 3 [--style heldout]``: write plans."""

from __future__ import annotations

import argparse
from pathlib import Path

from kelpod.evaluation.synth import generate, save


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("samples"))
    ap.add_argument("--n", type=int, default=3, help="number of plans")
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--degrade", type=float, default=0.5, help="degradation strength 0..1")
    ap.add_argument("--dpi", type=float, default=300.0)
    ap.add_argument(
        "--style", default="train", choices=["train", "heldout"], help="never train on heldout"
    )
    ap.add_argument("--clean", action="store_true", help="also write plan_clean.png, wall_mask.png")
    args = ap.parse_args(argv)
    for seed in range(args.seed, args.seed + args.n):
        plan = generate(seed, args.degrade, dpi=args.dpi, style=args.style)
        d = save(plan, args.out / f"seed_{seed:04d}", clean=args.clean)
        rooms = ", ".join(f"{r.name} {r.stamp_area_m2:.2f}" for r in plan.graph.rooms)
        print(f"{d}: {plan.image.shape[1]}x{plan.image.shape[0]} px, {rooms}")


if __name__ == "__main__":
    main()
