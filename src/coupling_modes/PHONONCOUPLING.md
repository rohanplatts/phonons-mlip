# Phonon mode--coupling comparison: DFT vs MLIPs (CBVN)


## Overview
This script compares the phonon coupling between a variety of instances of MLIPs to the gold standard DFT. for a single defect supercell.

Inputs:
- `CONTCAR_GS` and `CONTCAR_ES` (same atom ordering).
- One DFT `band.yaml` and a set of MLIP `band.yaml` files (Phonopy format).

Output: a text report that (i) selects DFT ``coupling'' modes from the GS$\to$ES displacement, (ii) scores each MLIP on several diagnostics, and (iii) produces a ranked list.

## What the script computes
**(1) GS$\to$ES mass-weighted displacement.**  
From the two CONTCARs we compute the minimum-image displacement in Cartesian coordinates, optionally remove the mass-weighted center-of-mass shift, and form the mass-weighted displacement vector:
$$
\Delta \mathbf r_i = \mathbf r_i^{\mathrm{ES}} - \mathbf r_i^{\mathrm{GS}},\qquad
\Delta \mathbf q = \bigl(\sqrt{m_1}\Delta \mathbf r_1, \ldots, \sqrt{m_N}\Delta \mathbf r_N\bigr)\in\mathbb{R}^{3N}.
$$

**(2) Coupling-mode weights from DFT eigenvectors.**  
Let $E(\mathbf q)\in\mathbb{C}^{3N\times 3N}$ be the DFT eigenvector matrix at a chosen $\mathbf q$ (columns normalized).  
Define per-mode amplitudes and projection power:
$$
a_m = \mathbf e_m^\dagger \Delta \mathbf q,\qquad
p_m = \frac{|a_m|^2}{\|\Delta \mathbf q\|^2}.
$$
The ``coupling modes'' are the smallest set of modes whose cumulative $p_m$ exceeds the threshold $\tau$.

For MLIP ranking we also define DFT per-mode weights
$$
w_m =
\begin{cases}
|a_m|^2, & \text{kind = `p`}\\
|\omega_m|\,|a_m|^2, & \text{kind = `S`}\\
|\omega_m|^2\,|a_m|^2, & \text{kind = `lambda`}
\end{cases}
$$
where $\omega_m$ is the DFT mode frequency. These weights decide which DFT modes ``matter'' most when scoring an MLIP.

**(3) GS$\to$ES score**  
For a matched DFT/MLIP $\mathbf q$-point, form the overlap matrix between eigenvector bases:
$$
O_{ij} = \bigl|\mathbf e_i^\dagger \tilde{\mathbf e}_j\bigr|^2\in[0,1].
$$
We compute the one-to-one assignment $\pi$ that maximizes $\sum_i O_{i,\pi(i)}$ (Hungarian algorithm).  
Using the DFT-defined weights $w_m$ (restricted to modes with $w_m>0$), we compute:
$$
E_{\mathrm{freq}} =
\sqrt{\frac{\sum_{m\in\mathcal V} w_m\,(\tilde\omega_{\pi(m)}-\omega_m)^2}{\sum_{m\in\mathcal V} w_m}},
\qquad
E_{\mathrm{vec}} =
\sqrt{\frac{\sum_{m\in\mathcal V} w_m\,(1-O_{m,\pi(m)})}{\sum_{m\in\mathcal V} w_m}},
$$
and the combined score
$$
\mathrm{Score} = E_{\mathrm{freq}} + \alpha\,E_{\mathrm{vec}}.
$$
Lower Score is better. I thought this would answer the question: `which MLIP best matches the DFT coupling modes eigenvalues and eigenvectors, and how well does it do that?`  
The different kinds of weights allow for different phrasings, for `kind=p` then the question is the above (where coupling modes refer to the vibrational modes containing the largest proportion of the mass weighted displacement vector).  
for `kind=lambda` the question is `How well does the MLIP reproduce the DFT modes that dominate the harmonic energetic cost of the GS to ES displacement?' and for `kind=S` the question is a mix. I need to confirm but I think it would be best suited for evaluating how accurate the PL spectra would be.

**(4) Secondary diagnostics (subspace agreement).**  
The report also prints:
- *AvgProjPowX*: compares the subspace spanned by the selected DFT coupling modes against the subspace spanned by the selected ML coupling modes, using principal angles. If $Q_d$ and $Q_m$ are orthonormal bases, with singular values $\sigma_\ell$ of $Q_d^\dagger Q_m$:
  $$
  X = \frac{1}{k_d}\sum_\ell \sigma_\ell^2,\quad (0\le X\le 1).
  $$
  I thought this score would answer the question: `How well do the strongly coupled phonons of <MLIP> match those of DFT?'
- *Coupling clusters*: clusters modes by frequency proximity (gap $>\Delta f_{\mathrm{cluster}}$ starts a new cluster), assigns each DFT cluster a weight (fraction of $\Delta\mathbf q$ captured), and checks whether ML frequencies in a window around each DFT cluster recover a similar subspace/weight. i thought this would answer the question `How well does the eigenspace of an <MLIP> frequency range match the displacements predicted by DFT in the same range?'


# Dictionary for `phonon_coupling_report_{i}.txt`

## Scores used in the report

- **Singular values $\sigma_i$ (report fields: $\sigma_{\min}$, $\sigma_{\mathrm{mean}}$)**  
  Given two subspaces with orthonormal bases $Q_A,Q_B$,
  $$
  C=Q_A\Herm Q_B,\qquad \{\sigma_i\}=\mathrm{svd}(C),\qquad 0\le\sigma_i\le 1.
  $$
  Interpretation: $\sigma_i\approx 1$ means aligned; $\sigma_i\approx 0$ means orthogonal.

- **Principal angles $\theta_i$ (report fields: $\theta_{\mathrm{mean}}$, $\theta_{\max}$)**  
  $$
  \theta_i=\arccos(\sigma_i)\times\frac{180}{\pi}\quad\text{(degrees)}.
  $$
  Interpretation: $0^\circ$ is aligned; $90^\circ$ is orthogonal.

- **$k_{\mathrm{DFT}}$ (`k_dft`)**  
  $k_{\mathrm{DFT}}=\dim\big(\mathrm{span}\{\mathbf e_m^{\mathrm{DFT}}:m\in S_\tau^{\mathrm{DFT}}\}\big)$.  
  Interpretation: dimension of the DFT coupling-mode subspace.

- **$k_{\mathrm{ML}}$ (`k_ml`)**  
  analogous dimension for the MLIP coupling-mode subspace.  
  Interpretation: if $k_{\mathrm{ML}}<k_{\mathrm{DFT}}$, the  subspace score \(X\) is capped.

- **$X$ (`AvgProjPowX`, printed as `X`)**  
  For coupling-mode subspaces $Q_{\mathrm{DFT}}$ and $Q_{\mathrm{ML}}$,
  $$
  X(\mathbf q)=\frac{1}{k_{\mathrm{DFT}}}\sum_{i=1}^{\min(k_{\mathrm{DFT}},k_{\mathrm{ML}})}\sigma_i^2\in[0,1].
  $$
  Interpretation: coupling-subspace agreement; closer to $1$ is better.

- **`X: mean/min/max`**  
  For multiple q-points: `mean`$=\langle X(\mathbf q)\rangle_{\mathbf q}$ and `min/max` are extrema across analyzed $\mathbf q$.  
  Interpretation: variability of subspace agreement over q (with $\Gamma$-only, these often coincide).

- **$X_{\mathrm{mean}}$ (`X_mean`)**  
  $X_{\mathrm{mean}}=\langle X(\mathbf q)\rangle_{\mathbf q}$.

- **`dQ score: mean/min`**  
  In the per-model header, (see below for `Score()`)
  $$
  \texttt{mean}=\langle \mathrm{Score}(\mathbf q)\rangle_{\mathbf q}=\mathrm{Score\_mean},
  \qquad
  \texttt{min}=\min_{\mathbf q}\mathrm{Score}(\mathbf q).
  $$

- **`L1w` and `L1w_mean`**
  $$
  \mathrm{L1w}(\mathbf q)=\sum_{C\in\mathcal R_\tau(\mathbf q)}\big|w_C^{\mathrm{ML}}(\mathbf q)-w_C^{\mathrm{DFT}}(\mathbf q)\big|.
  $$
  Interpretation: cluster-weight mismatch in DFT-important spectral regions; $0$ is perfect.

- **`L1w_mean`**  
  $\mathrm{L1w\_mean}=\langle \mathrm{L1w}(\mathbf q)\rangle_{\mathbf q}$.  
  Interpretation: average L1 cluster-weight mismatch across analyzed q-points.

- **`wtheta_mean`**  
  $\mathrm{w\theta\_mean}=\langle w\theta(\mathbf q)\rangle_{\mathbf q}$.  
  Interpretation: average DFT-weighted cluster misalignment angle (degrees).

- **`w(1-sigma2)_mean`**  
  $\langle w(1-\sigma^2)(\mathbf q)\rangle_{\mathbf q}$.  
  Interpretation: average cluster-level $\sigma^2$ misalignment score across q-points.

- **`wtheta` and `wtheta_mean`**  
  For each relevant cluster $C$, compute principal angles between $Q_C^{\mathrm{DFT}}$ and $Q_{W(C)}^{\mathrm{ML}}$ and take the mean angle $\overline{\theta}_C(\mathbf q)$.  
  Then
  $$
  w\theta(\mathbf q)=\sum_{C\in\mathcal R_\tau(\mathbf q)} w_C^{\mathrm{DFT}}(\mathbf q)\,\overline{\theta}_C(\mathbf q)\quad (\text{degrees}).
  $$
  Interpretation: DFT-weighted subspace misalignment inside the important frequency windows; smaller is better.

- **`w(1-sigma2)` and `w(1-sigma2)_mean`**  
  For each relevant cluster $C$, define
  $$
  \sigma_C^2(\mathbf q)=\frac{1}{\dim(Q_C^{\mathrm{DFT}})}\sum_i \sigma_i(C;\mathbf q)^2,\qquad
  w(1-\sigma^2)(\mathbf q)=\sum_{C\in\mathcal R_\tau(\mathbf q)} w_C^{\mathrm{DFT}}(\mathbf q)\big(1-\sigma_C^2(\mathbf q)\big).
  $$
  Interpretation: another cluster-level misalignment score; $0$ is best.

- **$O_{ij}$ (used inside `E_vec`)**
  $$
  O_{ij}(\mathbf q)=\left|\mathbf e_i^{\mathrm{DFT}}(\mathbf q)\Herm\mathbf e_j^{\mathrm{ML}}(\mathbf q)\right|^2\in[0,1].
  $$
  Interpretation: phase/sign-invariant similarity between individual modes.

- **$\pi$ (Hungarian assignment, used inside `E_freq`, `E_vec`)**
  $$
  \pi=\arg\max_{\text{one-to-one maps}}\sum_{i=1}^{3N} O_{i,\pi(i)}.
  $$
  Interpretation: matches MLIP modes to DFT modes without trusting mode indices (or frequencies)

- **`weight_kind`**  
  Selects nonnegative DFT importance weights $w_i$ for the targeted (mode-by-mode) RMS scores.
  $$
  w_i \propto |a_i|^2
  $$
  Interpretation: emphasizes modes that contribute to the GS$\to$ES displacement, or, depending on kind, emphasises higher frequency modes.

- **$\alpha$ (`alpha`)**  
  Tradeoff scalar in the final targeted score.  
  Interpretation: larger $\alpha$ penalizes eigenvector mismatch more strongly relative to frequency error. I settled for around 1.3.

- **`E_freq`**  
  With normalized weights $\tilde w_i=w_i/\sum_j w_j$ and assigned partner $\pi(i)$,
  $$
  E_{\mathrm{freq}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\left(\omega_{\pi(i)}^{\mathrm{ML}}(\mathbf q)-\omega_i^{\mathrm{DFT}}(\mathbf q)\right)^2}.
  $$
  Interpretation: weighted RMS frequency error on GS$\to$ES- DFT coupling modes; $0$ is best.

- **`E_freq_rel`**
  $$
  E_{\mathrm{freq,rel}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\left(\frac{\omega_{\pi(i)}^{\mathrm{ML}}-\omega_i^{\mathrm{DFT}}}{\max(|\omega_i^{\mathrm{DFT}}|,\varepsilon)}\right)^2}.
  $$
  Interpretation: weighted RMS *relative* frequency error (dimensionless).

- **`E_vec`**  
  Define $d_i(\mathbf q)=1-O_{i,\pi(i)}(\mathbf q)\in[0,1]$.  
  Then
  $$
  E_{\mathrm{vec}}(\mathbf q)=\sqrt{\sum_i \tilde w_i\,d_i(\mathbf q)^2}.
  $$
  Interpretation: weighted RMS displacement-pattern mismatch; $0$ is best.

- **`Score` and `Score_mean`**
  $$
  \mathrm{Score}(\mathbf q)=E_{\mathrm{freq}}(\mathbf q)+\alpha\,E_{\mathrm{vec}}(\mathbf q),
  \qquad
  \mathrm{Score\_mean}=\langle \mathrm{Score}(\mathbf q)\rangle_{\mathbf q}.
  $$
  Interpretation: used for ranking MLIPs (smaller is better).

- **`q-points used`**  
  $n_q=\text{number of q-points included in the report}$.  
  Interpretation: with `Gamma_only`=`True`, $n_q=1$, so ``mean/min/max'' coincide.
