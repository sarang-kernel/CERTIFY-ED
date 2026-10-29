# Planted-error corpus

`corpus/` in the repository holds 60 small exact-diagonalization results with one known error each
(wrong eigenvalue, unnormalised vector, swapped pairs, missing multiplet member, duplicated vector, wrong
coupling, wrong matrix entry, non-Hermitian producer matrix, NaN, mislabeled sector, subnormal entries,
wrong window count) on the Heisenberg chain (N = 8, 10), the J1-J2 chain and the transverse-field Ising
chain, plus one control per model. Each case is a producer bundle with an `expected.json` naming the
statements a correct checker must not certify, derived from the statement definitions
(`SCOPE_MATRIX.md`, `theory.md`), not from any checker's output.

```
python corpus/run_corpus.py                                      # CERTIFY-ED of this checkout
python corpus/run_corpus.py --checker "mychecker {case} --out {out}" --name mychecker
```

The runner scores any checker that writes a certificate or a flat `{statement: status}` object:
*detected*, *unsound* (a false statement certified), *silent* (not checked), *refused*, *conditional*,
*error*, and false alarms on the controls; it writes `corpus/results/<name>/report.md`. The run of
CERTIFY-ED 3.2.0.dev0 is committed under `corpus/results/certify-ed/` and summarised in the CHANGELOG.
`corpus/README.md` describes every error class, what a correct checker must say about it, the scoring
rules and how to submit a case.
