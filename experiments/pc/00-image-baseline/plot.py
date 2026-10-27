import matplotlib
matplotlib.use("Agg")                    # render to files only, never open a window
import matplotlib.pyplot as plt

def plot_sweep(rows, cfg, size, path, impl="gaussian_separable"):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))         # 2x3 grid of panels; size in inches
    for ax, pattern in zip(axes.flat, cfg["patterns"]):     # pair each panel with a pattern name
        for s_n in cfg["noise"]["sigmas"]:                  # one line per noise level
            xs, ys = [], []
            for r in rows:
                if (r["pattern"] == pattern and r["size"] == size and r["sigma_n"] == s_n
                        and (r["filter"] == "no_filter" or r["filter"].startswith(impl))):
                    xs.append(r["blur_sigma"])
                    ys.append(r["psnr"])
            ax.plot(xs, ys, marker="o", label=f"σn = {s_n}")
        ax.set_title(pattern)
        ax.set_xlabel("blur σ")
        ax.set_ylabel("PSNR [dB]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)              # legend in every panel
    fig.tight_layout()                                      # stop labels overlapping
    fig.savefig(path, dpi=150)
    plt.close(fig)