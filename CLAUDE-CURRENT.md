# CLAUDE-CURRENT.md

Current status and open items. Newest first; dates absolute.

## 2026-11-09 -- Python 3.10 dropped

CI on Python 3.10 (Ubuntu and macOS) failed at `import psi4` from 2026-10-18 on:
`undefined symbol: _ZN7einsums13random_engineE`. conda-forge rebuilt Psi4 against
`einsums` 1.1.6 on 2026-10-11 for Python 3.11+ only; the last 3.10 Psi4 build
(2026-08-08) was built against 1.1.5 but allows `<1.2`, so it gets 1.1.6 and fails.
Python 3.10 reached end of life in October 2026 (PEP 619). Minimum is now 3.11, on
branch `ci/drop-python-3.10`.

### TO-DO

- [ ] Optional: report the loose `einsums >=1.1.5,<1.2` pin on the old Psi4 3.10
      build to conda-forge's psi4 feedstock.

## 2026-09-10 -- Factored Lambda equations implemented

Both CCSD Lambda residuals now run in their factored form in `cc_wfn.py`, 11
contractions apiece behind eight intermediates, down from 45 singles terms and 32
doubles representatives. `_solve_lambda_SO` builds the six t-only intermediates
once before the iteration; only `G_om` and `G_ge` depend on Lambda and are rebuilt
each cycle. Same amplitudes as before to 10 digits, and faster
(H2O/6-31G* CCSD+Lambda: 3.22 s -> 2.33 s, T iteration unchanged in both).

- Derivation: `~/Documents/Notes/ccsd_derivation/main.tex` (content in `theory.tex`).
- Verification of the document: `~/Documents/Notes/ccsd_derivation/verify/`.
- Verification of the code: `test_028_CCSD_Lambda_SO.py`, which now carries the
  generated unfactored residuals as `_generated_lambda_singles` /
  `_generated_lambda_doubles` and compares elementwise on random complex tensors.
- Summary of the intermediates and which are shared: see the CCSD section of
  `CLAUDE.md`.

The T-side factored implementation was checked against the document's raw
Eqs. `ccsd_t1` / `ccsd_t2` on random complex tensors and matches; no change was
needed there.

Ten mutations of the Lambda layer (permutation-vs-factor-2, bra-ket inversions,
dropped prefactors, wrong swap axes, stale G dressing) were all killed by
`test_028`.

### TO-DO

- [ ] **Open question: naming the merged G.** Both G terms in the doubles read
      `G - t1*l1`, so the obvious move is to fold that into a redefined `G`. It is
      valid but not worth it: substituting the augmented `G` into the singles leaves
      `sum <oi||ga> t_o^e t_m^g l_m^e` at twice the size of the single `W~_eima`
      term available to absorb it, and both G terms want that same term, so the
      merge adds terms rather than removing any. Kept as an explicit group
      (`G_oo_dressed`, `G_vv_dressed`).
- [ ] Minor, document only: most singles intermediates in `theory.tex` carry no
      `\label`, so they cannot be cross-referenced. Add labels if the document grows
      references to them.

### Next

- CCSD-level AATs by finite difference, which is what the Lambda amplitudes are for.

### Not doing

- Spin-adapted CCSD -- still downstream of the AATs, unchanged.
- `aats.py` normalisation `N = 1/sqrt(t0^2 + <t1|t1> + (1/4)<t2|t2>)` is CI-style
  intermediate normalisation and does **not** apply to CC. The FD AAT code will need
  a CC-specific branch. See the CCSD section of `CLAUDE.md`.
