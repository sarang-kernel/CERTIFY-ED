"""N-11: the nine-coupling spin-1/2 crown of Kranitz & Penc (arXiv:2601.07800, Appendix D).

Certified enclosure lower <= lambda_min(H_0) <= upper at the nine printed couplings of Eq. (D4), in exact
arithmetic (certify_ed.apps.pyrochlore.nine_coupling.certify_nine), compared with the paper's e_LB = -0.549832(8)
and with the floating-point reproduction -0.54983289 of the review (finding B2-1). With --compare, floating-point
Lanczos values are also recorded for the two other readings of the printed digits ("truncated", and the review's
reading with c4 = 0.0000415).

usage: python experiments/n11_crown_nine.py [appended|truncated] [--compare] [--cap-gib X]
"""
import sys
from fractions import Fraction

from harness import Run
from portable import cap_memory, peak_rss_mb
from certify_ed.apps.pyrochlore import nine_coupling as nc


def main(argv):
    reading = next((a for a in argv if a in ("appended", "truncated")), "appended")
    cap_gib = float(argv[argv.index("--cap-gib") + 1]) if "--cap-gib" in argv else 5.0
    mech = cap_memory(int(cap_gib * 2 ** 30))
    with Run("n11_crown_nine") as run:
        run.record("memory_cap", {"gib": cap_gib, "mechanism": mech})
        run.record("orbit_sizes", nc.check_orbit_table())
        run.record("printed", {f"c{k}": v for k, v in nc.PRINTED.items()})
        run.log(f"memory cap {cap_gib} GiB: {mech}; reading {reading}")
        if "--compare" in argv:
            cmp = {}
            free = nc.free_couplings("appended")
            free_review = dict(free, **{4: Fraction(415, 10 ** 7)})
            for name, f in (("appended", None), ("truncated", nc.free_couplings("truncated")),
                            ("review_c4_0.0000415", free_review)):
                w, _, _ = nc.estimate(nc.couplings(free=f) if f is not None else nc.couplings(), k=2)
                cmp[name] = [float(x) for x in w]
                run.log(f"floating point, reading {name}: lowest two {w[0]:.12f} {w[1]:.12f}")
            run.record("float_readings", cmp)
        r = nc.certify_nine(reading=reading, log=run.log)
        pub = nc.printed_value(nc.PUBLISHED_BOUND, "appended")
        r["published_bound"] = nc.PUBLISHED_BOUND
        r["published_appended"] = float(pub)
        r["review_float"] = nc.REVIEW_FLOAT
        lb = Fraction(r["lower_bound_exact"])
        ub = Fraction(r["upper_bound_exact"])
        r["lower_minus_published"] = float(lb - pub)
        r["review_float_in_enclosure"] = bool(lb <= Fraction(nc.REVIEW_FLOAT) <= ub)
        r["reduced_model_bound"] = -0.5501582988212038      # spin_half record, 3.2.1 run (E9)
        r["gain_over_reduced_model"] = float(lb - Fraction(-0.5501582988212038))
        r["peak_rss_mb"] = peak_rss_mb()
        run.record("nine_coupling", r)
        run.log(f"certified {r['lower_bound_rounded_down']:.9f} <= lambda_min <= {r['upper_bound_rounded_up']:.9f}; "
                f"published {nc.PUBLISHED_BOUND}; peak {r['peak_rss_mb']:.0f} MiB")


if __name__ == "__main__":
    main(sys.argv[1:])
