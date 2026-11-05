"""Plots the training curve of a run as PSNR per epoch: runs/<name>/curve.png."""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = Path(__file__).parent / "runs"


def psnr(mse):
    return 10 * math.log10(1.0 / mse)  # peak 1, as in 00


def main(name="depth13"):
    with open(RUNS / name / "curve.csv") as f:
        rows = list(csv.DictReader(f))
    epoch = [int(r["epoch"]) for r in rows]
    train = [psnr(float(r["train_mse"])) for r in rows]
    valid = [psnr(float(r["valid_mse"])) for r in rows]
    noisy = psnr(0.1 ** 2)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(epoch, train, color="#2a78d6", lw=2, label="Training patches")
    ax.plot(epoch, valid, color="#eb6834", lw=2, label="Validation crops")
    ax.axhline(noisy, color="#52514e", lw=1, ls="--", label="Noisy input")
    for y, color in ((train[-1], "#2a78d6"), (valid[-1], "#eb6834")):
        ax.plot(epoch[-1], y, "o", ms=6, color=color)
        ax.annotate(f"{y:.2f} dB", (epoch[-1], y), xytext=(6, 0),
                    textcoords="offset points", va="center", color="#0b0b0b")
    ax.set_xlabel("Epoch", color="#52514e")
    ax.set_ylabel("PSNR (dB)", color="#52514e")
    ax.set_title(f"Training curve, {name}", color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, epoch[-1] + 12)
    fig.tight_layout()
    fig.savefig(RUNS / name / "curve.png", dpi=150)
    print(f"final train {train[-1]:.2f} dB, valid {valid[-1]:.2f} dB, noisy {noisy:.2f} dB")


if __name__ == "__main__":
    main()