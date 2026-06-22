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

## CISD AAT component cross-check (`analytic_aats.compute_CISD_AATs`)

This is an independent re-derivation of every CISD AAT component (HF, S0, 0S, SS,
DS, SD, DD, and the normalization term). The production code writes the
overlap-derivative contributions using the compact half-overlap-derivative term
`half_S_core_a`; this alternative writes them out in the explicit
`-0.5*S_core[a] + half_S_core_a.T` form. The two formulations are algebraically
equivalent, so the `_test` arrays should match the production arrays — the
`assert`s at the end verify this. It is written for the `non-canonical`,
no-frozen-core case.

To use it: declare the `_test` accumulators just before the `for beta in range(3)`
loop, paste the block below inside the loop (after the production AAT terms), and
the assertions validate the production result element by element.

```python
# Declare alongside the production AAT component arrays.
AAT_HF_test = np.zeros((natom * 3, 3))
AAT_S0_test = np.zeros((natom * 3, 3))
AAT_0S_test = np.zeros((natom * 3, 3))
AAT_SS_test = np.zeros((natom * 3, 3))
AAT_DS_test = np.zeros((natom * 3, 3))
AAT_SD_test = np.zeros((natom * 3, 3))
AAT_DD_test = np.zeros((natom * 3, 3))
AAT_Norm_test = np.zeros((natom * 3, 3))
```

```python
if orbitals == 'non-canonical' and self.parameters['freeze_core'] == False:
    # Computing the Hartree-Fock term of the AAT.
    AAT_HF_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("em,em", U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

    # Singles/Reference terms.
    AAT_S0_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dt1_dR, U_H[beta][v_,o_])

    AAT_S0_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ei,ea", t1, U_H[beta][v_,o_], -0.5*S_core[a][v_,v_] + half_S_core_a[v_,v_].T)
    AAT_S0_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,am,im", t1, U_H[beta][v_,o], -0.5*S_core[a][o_,o] + half_S_core_a[o,o_].T)

    # Reference/Singles terms.
    AAT_0S_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ai", dT1_dH[beta], U_R[v_,o_] + half_S_core_a[o_,v_].T)

    # Singles/Singles terms.
    AAT_SS_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ia", dt1_dR, dT1_dH[beta])

    AAT_SS_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,ae,ie", dT1_dH[beta], -0.5*S_core[a][v_,v_] + half_S_core_a[v_,v_].T, t1)
    AAT_SS_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,mi,ma", dT1_dH[beta], -0.5*S_core[a][o_,o_] + half_S_core_a[o_,o_].T, t1)

    AAT_SS_test[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,ia,kc,kc", t1, U_H[beta][o_,v_], U_R[o_,v_] + half_S_core_a[v_,o_].T, t1)
    AAT_SS_test[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,em,em,ia", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core_a[o,v_].T, t1)
    AAT_SS_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,ei,ma", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)
    AAT_SS_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,am,ie", t1, U_H[beta][v_,o], U_R[v_,o] + half_S_core_a[o,v_].T, t1)
    AAT_SS_test[lambda_alpha][beta] += N**2 * 4 * oe.contract("ia,em,ai,me", t1, U_H[beta][v_,o_], U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)

    # Doubles/Singles terms.
    AAT_DS_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2*dt2_dR - dt2_dR.swapaxes(2,3), U_H[beta][v_,o_], t1)

    AAT_DS_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dT1_dH[beta], U_R[o_,v_] + half_S_core_a[v_,o_].T, 2*t2 - t2.swapaxes(2,3))

    AAT_DS_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ia,em,km,kiea", t1, U_H[beta][v_,o], -0.5*S_core[a][o_,o] + half_S_core_a[o,o_].T, 2*t2 - t2.swapaxes(2,3))
    AAT_DS_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,em,ec,imac", t1, U_H[beta][v_,o_], -0.5*S_core[a][v_,v_] + half_S_core_a[v_,v_].T, 2*t2 - t2.swapaxes(2,3))

    # Singles/Doubles terms.
    AAT_SD_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ia,kc,ikac", dt1_dR, U_H[beta][o_,v_], 2*t2 - t2.swapaxes(2,3))

    AAT_SD_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,bj,ia", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), U_R[v_,o_] + half_S_core_a[o_,v_].T, t1)

    # Doubles/Doubles terms.
    AAT_DD_test[lambda_alpha][beta] += N**2 * oe.contract("ijab,ijab", 2*dt2_dR - dt2_dR.swapaxes(2,3), dT2_dH[beta])

    AAT_DD_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,kjab,ki", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, -0.5*S_core[a][o_, o_] + half_S_core_a[o_, o_].T)
    AAT_DD_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijcb,ac", 2*dT2_dH[beta] - dT2_dH[beta].swapaxes(2,3), t2, -0.5*S_core[a][v_, v_] + half_S_core_a[v_, v_].T)

    AAT_DD_test[lambda_alpha][beta] += N**2 * 2 * oe.contract("ijab,ijab,em,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)
    AAT_DD_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,imab,ej,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o_], U_R[v_, o_] + half_S_core_a[o_, v_].T)
    AAT_DD_test[lambda_alpha][beta] -= N**2 * 2 * oe.contract("ijab,ijae,bm,em", t2, 2*t2 - t2.swapaxes(2,3), U_H[beta][v_, o], U_R[v_, o] + half_S_core_a[o, v_].T)

    # Adding terms for full normalization.
    if normalization == 'full':
        AAT_Norm_test[lambda_alpha][beta] += N * N_R * 1 * oe.contract("ijab,ijab", 2*t2 - t2.swapaxes(2,3), dT2_dH[beta])

        AAT_Norm_test[lambda_alpha][beta] += N * N_R * 4 * oe.contract("ia,ai", t1, U_H[beta][v_, o_])
        AAT_Norm_test[lambda_alpha][beta] += N * N_R * 2 * oe.contract("ia,ia", t1, dT1_dH[beta])
        AAT_Norm_test[lambda_alpha][beta] += N * N_R * 4 * oe.contract("ijab,bj,ia", 2*t2 - t2.swapaxes(2,3), U_H[beta][v_,o_], t1)

    assert AAT_HF_test.all() == AAT_HF.all()
    assert AAT_S0_test.all() == AAT_S0.all()
    assert AAT_0S_test.all() == AAT_0S.all()
    assert AAT_SS_test.all() == AAT_SS.all()
    assert AAT_DS_test.all() == AAT_DS.all()
    assert AAT_SD_test.all() == AAT_SD.all()
    assert AAT_DD_test.all() == AAT_DD.all()
    assert AAT_Norm_test.all() == AAT_Norm.all()
```
