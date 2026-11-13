"""N-12: size, memory and time of the honeycomb criterion of Pomata and Wei (arXiv:1911.01410, Method III) with the
eta machinery of certify-ed (certify_ed.apps.aklt.eta.certify_eta), before anything is attempted.

The geometry is lattices.honeycomb_method3(); its dimensions are compared with Table I of the paper (rank E = 2^12,
rank E ^ F = 2^18, dim H_L = dim H_R = 2^10, dim H_C = 2^6, D_tot = 2^26). The bordered blocks are counted with
estimate.plan_eta, whose largest blocks are compared with those that certify_eta logged for the other lattices in the
E10 run of the 3.2.1 evidence run (code/certify-ed/results/run_full_20261105/E10.log: star 70, cross 1584, decorated
honeycomb 673, decorated square 4034 and 3959 with the mirror). They agree for the star, the cross and the decorated
honeycomb; for the decorated square plan_eta gives 3997 rows against the logged 4034, so its block sizes are an
estimate, not a reproduction, wherever the site symmetries enter. Seconds per n^3 of the exact re-check are calibrated on the cross and
the decorated square of that log (time of the counting step = total minus the time before the blocks), and the larger
of the two values is used.

usage: python experiments/n12_honeycomb_estimate.py
"""
from fractions import Fraction
from math import comb

from harness import Run
from portable import peak_rss_mb
from certify_ed.apps.aklt import lattices
from certify_ed.apps.aklt.estimate import plan_eta, EXACT_BYTES_PER_ENTRY

# E10.log of the 3.2.1 run: seconds at the end of the count and at the start of the blocks, largest blocks logged
E10_LOG = {"star": {"largest": 70}, "cross": {"largest": 1584, "end_s": 106, "blocks_start_s": 3},
           "decorated_honeycomb": {"largest": 673},
           "decorated_square": {"largest": 4034, "end_s": 2368, "blocks_start_s": 10}}
PAPER_TABLE_I = {"Gamma_i": 2 ** 12, "union": 2 ** 18, "L": 2 ** 10, "C": 2 ** 6, "R": 2 ** 10, "dim_V": 2 ** 26}
PAPER_ETA = "0.1445124916"
PAPER_GAMMAS = {"gamma_p": "0.3130520508", "gamma_2p": "0.02571076873", "gamma_4p": "0.7784203312"}
PAPER_FINAL_EQUATION = "0.002660333395"
PAPER_CONCLUSION = "0.002550333395"


def main():
    with Run("n12_honeycomb_estimate") as run:
        checks = {}
        for name in ("star", "cross", "decorated_honeycomb", "decorated_square"):
            g, L, C, R = getattr(lattices, name)()
            sig = [lattices.decorated_square_reflection(g)] if name == "decorated_square" else ()
            p = plan_eta(g, L, C, R, sigmas=sig)
            checks[name] = {"largest_block_rows": p["largest_block_rows"], "e10_log_largest": E10_LOG[name]["largest"],
                            "sum_n3": p["sum_n3"], "site_symmetry_group_used": p["site_symmetry_group_used"]}
            if "end_s" in E10_LOG[name]:
                sec = E10_LOG[name]["end_s"] - E10_LOG[name]["blocks_start_s"]
                checks[name]["count_seconds"] = sec
                checks[name]["seconds_per_n3"] = sec / p["sum_n3"]
            run.log(f"{name}: largest block {p['largest_block_rows']} (E10 log {E10_LOG[name]['largest']}), "
                    f"sum n^3 {p['sum_n3']:.4g}")
        spn = max(c["seconds_per_n3"] for c in checks.values() if "seconds_per_n3" in c)
        run.record("calibration", {"lattices": checks, "seconds_per_n3_used": spn,
                                   "bytes_per_entry_used": EXACT_BYTES_PER_ENTRY})
        g, L, C, R = lattices.honeycomb_method3()
        h = plan_eta(g, L, C, R, seconds_per_n3=spn)
        geo = {k: h["reduced"].get(k, h.get(k)) for k in ("Gamma_i", "union", "L", "C", "R")}
        geo["dim_V"] = h["dim_V"]
        h["paper_table_I"] = PAPER_TABLE_I
        h["geometry_matches_table_I"] = geo == PAPER_TABLE_I
        # a reduction by total spin (not implemented): the bordered block of the spin-S multiplicity space has
        # 2 (C(22, 11 - S) - C(22, 10 - S)) rows
        mult = {S: comb(22, 11 - S) - (comb(22, 10 - S) if S <= 10 else 0) for S in range(0, 12)}
        Smax = max(mult, key=mult.get)
        h["su2_reduced_largest_rows"] = {"S": Smax, "rows": 2 * mult[Smax]}
        h["years_one_core_all_blocks"] = h["all_blocks_exact_seconds"] / (365.25 * 86400)
        run.record("honeycomb_method3", h)
        run.log(f"honeycomb Method III: dim V {h['dim_V']}, bordered rows {h['bordered_rows']}, largest block "
                f"{h['largest_block_rows']} rows, float copy {h['largest_block']['dense_float64_copy_GiB']:.4g} GiB, exact "
                f"{h['largest_block']['exact_GiB']:.4g} GiB, all blocks {h['all_blocks_exact_seconds']:.4g} s; Table I match "
                f"{h['geometry_matches_table_I']}")
        eta = Fraction(PAPER_ETA)
        prod = Fraction(PAPER_GAMMAS["gamma_2p"]) * Fraction(PAPER_GAMMAS["gamma_4p"]) * (1 - 6 * eta)
        run.record("gap_constants", {"printed": PAPER_GAMMAS, "eta": PAPER_ETA, "product_exact": f"{prod.numerator}/{prod.denominator}",
                                     "product": float(prod), "final_equation": PAPER_FINAL_EQUATION,
                                     "conclusion": PAPER_CONCLUSION,
                                     "final_equation_minus_product": float(Fraction(PAPER_FINAL_EQUATION) - prod),
                                     "conclusion_minus_product": float(Fraction(PAPER_CONCLUSION) - prod)})
        run.log(f"gamma_2p gamma_4p (1 - 6 eta) = {float(prod):.13g}; final equation {PAPER_FINAL_EQUATION}, "
                f"conclusion {PAPER_CONCLUSION}")
        run.record("lemm_sandvik_wang_cluster", {"sites": 36, "spin": "3/2", "states": 4 ** 36})
        run.record("peak_rss_mb", peak_rss_mb())


if __name__ == "__main__":
    main()
