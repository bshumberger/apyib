# Developer Notes

Diagnostic snippets that are useful when debugging or verifying parts of `apyib`
but that are not part of the normal calculation path. Paste them into the
indicated function temporarily when you need them; they are kept here rather than
as commented-out code in the source.

## SCF energy and density checks (`hf_wfn.solve_SCF`)

These run inside `solve_SCF`, just before `return E_SCF, self.C`, where `C`, `F`,
`D`, `H_core`, and `H` are all in scope.

**Cross-check the SCF energy three ways.** The AO→MO transformed energy and the
AO-density-based energy should each equal the converged `E_SCF` (plus the nuclear
repulsion energy).

```python
# AO -> MO transformed SCF energy.
H_core_MO = oe.contract('ip,ij,jq->pq', np.conjugate(C), H.T + H.V, C)
F_MO = oe.contract('ip,ij,jq->pq', np.conjugate(C), F, C)
E1 = 0.0
for i in range(0, self.ndocc):
    E1 += H_core_MO[i][i] + F_MO[i][i]
print('AO to MO Transformed Energy:', E1 + self.H.E_nuc)

# AO density-based SCF energy.
E_SCF1 = oe.contract('vu,uv->', D, H_core + F)
print('AO Density-Based Energy:', E_SCF1 + self.H.E_nuc)
```

**Inspect the density and MO coefficients.** The AO density should equal the
occupied-block outer product `C_occ @ C_occ^†`.

```python
print('D')
print(D, '\n')
print('C_occ @ C_occ^H')
print(C[0:self.nbf, 0:self.ndocc] @ np.conjugate(np.transpose(C)[0:self.ndocc, 0:self.nbf]), '\n')
```

**MO-basis density** — `C^† S D S C` should be diagonal with the orbital
occupations on the diagonal.

```python
print(np.conjugate(np.transpose(C)) @ H.S @ D @ H.S @ C)
```

**Idempotency of the AO density** — at convergence `D = D S D`, so this should be
(numerically) zero.

```python
print(D - (D @ H.S @ D))
```
