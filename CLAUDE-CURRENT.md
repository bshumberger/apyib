# CLAUDE-CURRENT.md

Current status and open items. Newest first; dates absolute.

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
