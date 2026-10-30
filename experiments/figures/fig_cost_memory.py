"""E2 fig_cost_memory: certifier wall time and peak memory against block dimension on two stacks.

Reads e2_cost_*/data.json of the evidence run (--results; the Windows 3.1.0 run) and of the second-stack
run (--results-b; the Linux 3.0.0 run). Each point is one E2 block (chain U(1) blocks, chain k = 0 blocks,
4 x 4 square blocks); filled when S3 (completeness) was CERTIFIED, open when S3 was NOT_CHECKED because the
factorization did not fit under the memory cap. The horizontal line is the cap (settings.memory_cap_GiB
of the evidence run). Time is `certifier.total`, memory `certifier.maxrss_mb` (MiB); both from the
record, not recomputed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

NAME = "fig_cost_memory"
FAMILY = {("chain", "u1"): ("chain, U(1) block", "o"), ("chain", "k0"): ("chain, k = 0 block", "s"),
          ("square", "u1"): ("4 × 4 square, U(1) block", "^")}


def family(r):
    return (r["cfg"]["model"], r["cfg"]["block"])


def points(data):
    out = []
    for r in data["rows"]:
        c = r.get("certifier") or {}
        if "statements" not in c:
            continue
        out.append({"dim": r["producer"]["dim"], "label": r["label"], "fam": family(r),
                    "seconds": c["total"], "mib": c["maxrss_mb"],
                    "s3": c["statements"]["S3.completeness"]["status"]})
    return out


def draw(dA, dB, nameA, nameB):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    C.style()
    pa, pb = points(dA), points(dB)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw=dict(wspace=0.3, left=0.09, right=0.99,
                                                                        bottom=0.17, top=0.88))
    cap_mib = dA["settings"]["memory_cap_GiB"] * 1024
    col = {nameA: C.ORANGE, nameB: C.BLUE}
    for pts, name in ((pb, nameB), (pa, nameA)):
        for p in pts:
            mk = FAMILY[p["fam"]][1]
            filled = p["s3"] == "CERTIFIED"
            kw = dict(marker=mk, ms=4.5, ls="none", color=col[name], mfc=col[name] if filled else "white",
                      mew=0.9, zorder=3)
            axes[0].plot(p["dim"], p["seconds"], **kw)
            axes[1].plot(p["dim"], p["mib"], **kw)
    axes[1].axhline(cap_mib, color=C.INK2, lw=0.7, ls="--")
    axes[1].text(700, cap_mib * 1.12, f"memory cap {dA['settings']['memory_cap_GiB']:g} GiB", fontsize=C.SIZES[2],
                 color=C.INK2, va="bottom")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("block dimension")
        ax.margins(x=0.08, y=0.12)
    axes[0].set_ylabel("certifier wall time (s)")
    axes[1].set_ylabel("certifier peak memory (MiB)")
    axes[0].set_title("the same 11 blocks certified on two stacks")
    axes[1].set_title("memory decides whether S3 runs")
    axes[0].set_yticks([0.1, 1, 10, 100, 1000])
    axes[0].set_yticklabels(["0.1", "1", "10", "100", "1000"])
    axes[1].set_yticks([100, 1000])
    axes[1].set_yticklabels(["100", "1000"])
    for ax in axes:
        ax.set_xticks([1000, 10000, 100000])
        ax.set_xticklabels(["1k", "10k", "100k"])
    C.panel_letter(axes[0], "a", dx=-0.22)
    C.panel_letter(axes[1], "b", dx=-0.22)
    h1 = [Line2D([], [], marker="o", ls="none", color=col[nameA], mfc=col[nameA], label=nameA.split(" (")[0]),
          Line2D([], [], marker="o", ls="none", color=col[nameB], mfc=col[nameB], label=nameB.split(" (")[0])]
    h2 = [Line2D([], [], marker=mk, ls="none", color=C.INK2, mfc=C.INK2, label=lab) for lab, mk in FAMILY.values()]
    h3 = [Line2D([], [], marker="o", ls="none", color=C.INK2, mfc=C.INK2, label="S3 CERTIFIED"),
          Line2D([], [], marker="o", ls="none", color=C.INK2, mfc="white", label="S3 NOT_CHECKED (cap)")]
    axes[0].legend(handles=h1 + h3, loc="upper left", fontsize=C.SIZES[2], handletextpad=0.3)
    axes[1].legend(handles=h2, loc="lower right", fontsize=C.SIZES[2], handletextpad=0.3)
    numbers = {}
    for pts, name in ((pa, nameA), (pb, nameB)):
        cert = [p for p in pts if p["s3"] == "CERTIFIED"]
        numbers[name] = {"blocks": len(pts), "S3_certified": len(cert),
                         "S3_not_checked": sum(p["s3"] == "NOT_CHECKED" for p in pts),
                         "max_dim_S3_certified": max(p["dim"] for p in cert) if cert else None,
                         "max_seconds": max(p["seconds"] for p in pts), "max_mib": max(p["mib"] for p in pts),
                         "largest_dim": max(p["dim"] for p in pts)}
    numbers["memory_cap_GiB"] = dA["settings"]["memory_cap_GiB"]
    return fig, numbers


def main():
    a = C.parse_args(__doc__.splitlines()[0], needs=("e2_cost",))
    dA, recA = C.load_record(a.results, "e2_cost")
    if not a.results_b or C.latest_record(a.results_b, "e2_cost") is None:
        C.absent("second-stack run (--results-b / CERTIFY_RESULTS_B) with an e2_cost record")
    dB, recB = C.load_record(a.results_b, "e2_cost")
    nameA = "Windows, certify-ed 3.1.0 (" + os.path.basename(os.path.normpath(a.results)) + ")"
    nameB = "Linux, certify-ed 3.0.0 (" + os.path.basename(os.path.normpath(a.results_b)) + ")"
    fig, numbers = draw(dA, dB, nameA, nameB)
    numbers["record_A"], numbers["record_B"] = recA, recB
    findings = C.check_layout(fig)
    paths = C.save(fig, a.out, NAME, a.png_only)
    C.write_numbers(a.out, NAME, numbers)
    for f in findings:
        print("layout:", f)
    print(numbers)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
