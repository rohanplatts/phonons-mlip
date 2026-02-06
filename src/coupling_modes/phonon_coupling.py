from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray
import yaml

ndarray_realFloats = ndarray  # expected float dtype
ndarray_complex = ndarray  # expected complex dtype

eps = 1e-12


@dataclass(frozen=True)
class Structure:
    """Container for crystal structure data.

    Args:
        lattice (ndarray_realFloats): (3,3) lattice vectors in Å.
        frac (ndarray_realFloats): (N,3) fractional coordinates.
        elements (List[str]): Element symbols in order.
        counts (List[int]): Atom counts per element.

    Returns:
        None
    """

    lattice: ndarray_realFloats  # (3,3) rows are lattice vectors in Å
    frac: ndarray_realFloats  # (N,3) fractional coordinates
    elements: List[str]
    counts: List[int]


@dataclass(frozen=True)
class BandData:
    """Phonon band structure data container.

    Args:
        natom (int): Number of atoms in the unit cell.
        masses (ndarray_realFloats): (N,) atomic masses.
        q_positions (ndarray_realFloats): (nq,3) q-point positions.
        frequencies (ndarray_realFloats): (nq,3N) phonon frequencies.
        eigenvectors (ndarray_complex): (nq,3N,N,3) phonon eigenvectors.

    Returns:
        None
    """

    natom: int
    masses: ndarray_realFloats  # (N,)
    q_positions: ndarray_realFloats  # (nq,3)
    frequencies: ndarray_realFloats  # (nq,3N)
    eigenvectors: ndarray_complex  # (nq,3N,N,3) complex128

    @property
    def nmodes(self) -> int:
        """Return number of phonon modes per q-point.

        Args:
            None

        Returns:
            int: Number of modes (second axis of frequencies).
        """

        return int(self.frequencies.shape[1])
    # frequencies are shaped like (number of q points, number of modes) (should be 3N haha)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> BandData:
        """Load phonon band data from a phonopy-style band.yaml.

        Args:
            path (Union[str, Path]): Path to band.yaml file.

        Returns:
            BandData: Parsed phonon band data.
        """

        p = Path(path)
        data = yaml.safe_load(p.read_text())
        # loads yaml file like a dictionary
        # would probably be better to use the phonopy.load however i stuck with this.

        phonon = data.get("phonon", None)
        if not phonon: # error catching to be precise about the phonopy structure requirement.
            raise ValueError(f"band.yaml missing/invalid 'phonon' list: {p}")

        if "natom" not in data:
            raise ValueError(f"band.yaml missing 'natom': {p}")
        natom = int(data["natom"])

        masses_list = _extract_masses(data, natom) # masses are stored per point (key) which has three values
        # one being its mass other being its symbol and the other being its coordinate.
        masses = np.asarray(masses_list, dtype=float) # preparing it to be diagonalised for mass 
        #weighted displacement vector

        # Accumulate per-q data; shapes after stacking:
        # q_positions -> (nq,3), freqs_all -> (nq,3N), eigs_all -> (nq,3N,N,3)
        q_positions: List[List[float]] = []
        freqs_all: List[List[float]] = []
        eigs_all: List[List[List[List[complex]]]] = [] # highly embodied since eig vectors themselves are list[list[complex]] 
        # and are per frequency which is per q point 

        nmodes: Optional[int] = None
        for ph in phonon:
            q = ph.get("q-position", None)
            band = ph.get("band", None)
            if q is None or band is None:
                raise ValueError(f"Invalid phonon entry (missing q-position/band) in {p}")
            if not isinstance(band, list) or not band:
                raise ValueError(f"Invalid band list at q-position in {p}")

            if nmodes is None:
                nmodes = len(band)
            elif len(band) != nmodes:
                raise ValueError(f"Inconsistent number of modes across q-points in {p}")

            # q-position is length-3 vector
            q_positions.append([float(q[0]), float(q[1]), float(q[2])])
            # convert these to floats as later we will be using l1 euclidean norm to qpoint match the mlip phonopy.yml
            # and the dft.

            freqs_q: List[float] = []  # length = nmodes (3N)
            eigs_q: List[List[List[complex]]] = []  # length = nmodes, each (N,3)
            for b in band:
                f = b.get("frequency", None)
                if f is None:
                    raise ValueError(f"Missing frequency in {p}")
                freqs_q.append(float(f))

                ev = b.get("eigenvector", None)
                if ev is None:
                    raise ValueError(f"Missing eigenvector in {p}")
                eigs_q.append(_parse_eigenvector(ev, natom))

            freqs_all.append(freqs_q)
            eigs_all.append(eigs_q)

        if nmodes is None:
            raise ValueError(f"No modes parsed from {p}")
        if nmodes != 3 * natom:
            raise ValueError(f"Expected nmodes==3N ({3*natom}) but got {nmodes} in {p}")

        # Final stacked arrays with fixed shapes
        q_positions_a = np.asarray(q_positions, dtype=float)  # (nq,3)
        freqs_a = np.asarray(freqs_all, dtype=float)  # (nq,3N)

        # Build eigenvector array explicitly to ensure shape (nq,3N,N,3)
        eigs = np.empty((len(phonon), nmodes, natom, 3), dtype=np.complex128)
        for iq in range(len(phonon)):
            for im in range(nmodes):
                eigs[iq, im, :, :] = np.asarray(eigs_all[iq][im], dtype=np.complex128)

        return cls( # cls is default class method is called on, we pass this info
            # through to return an instance of the BandData class.
            natom=natom,
            masses=masses,
            q_positions=q_positions_a,
            frequencies=freqs_a,
            eigenvectors=eigs,
        )

    def E(self, q_idx: int, normalize: bool = True) -> ndarray_complex:
        """Return eigenvector matrix at the given q-point.

        Args:
            q_idx (int): Index of the q-point.
            normalize (bool): Whether to normalize eigenvectors.

        Returns:
            ndarray_complex: (3N,3N) eigenvector matrix with columns as modes.
        """

        # Bounds check for q-point index
        if not (0 <= q_idx < self.q_positions.shape[0]):
            raise IndexError(f"q_idx out of range: {q_idx}")
        ev = self.eigenvectors[q_idx]  # (3N,N,3)
        # Flatten (N,3) per mode into (3N,) and stack as columns
        E = ev.reshape(self.nmodes, 3 * self.natom).T.copy()  # (3N,3N), cols are eigenvectors
        if normalize:
            norms = np.linalg.norm(E, axis=0)
            if np.any(norms <= eps):
                bad = np.where(norms <= eps)[0][:10].tolist()
                raise ValueError(f"Zero/near-zero eigenvector norm at modes {bad} (q_idx={q_idx})")
            E /= norms
        return E


@dataclass
class DFTCache:
    """Cached DFT-derived quantities for comparisons.

    Args:
        dq_flat (ndarray_realFloats): (3N,) mass-weighted displacement vector.
        dq_norm2 (float): Squared norm of dq_flat.
        masses (ndarray_realFloats): (N,) atomic masses.
        dft_path (str): Source band.yaml path.
        q_indices (List[int]): Selected DFT q-point indices.
        q_positions (ndarray_realFloats): (len(q_indices),3) q-point positions.
        freqs_by_q (List[ndarray_realFloats]): Per-q frequencies arrays.
        E_by_q (List[ndarray_complex]): Per-q eigenvector matrices.
        AvgProjPowX_by_q (List[Dict[str, Any]]): AvgProjPowX selection artifacts per q.
        clusters_by_q (List[List[List[int]]]): Per-q clusters of mode indices.
        cluster_ranges_by_q (List[List[Tuple[float, float]]]): Per-q (fmin,fmax) per cluster.
        w_dft_by_q (List[ndarray_realFloats]): Per-q DFT cluster weights.
        Q_cluster_by_q (List[List[ndarray_complex]]): Per-q orth bases per cluster.

    Returns:
        None
    """

    dq_flat: ndarray_realFloats  # (3N,)
    dq_norm2: float
    masses: ndarray_realFloats  # (N,)
    dft_path: str
    q_indices: List[int]
    q_positions: ndarray_realFloats  # (len(q_indices),3)

    freqs_by_q: List[ndarray_realFloats]  # each (3N,)
    E_by_q: List[ndarray_complex]  # each (3N,3N)
    AvgProjPowX_by_q: List[Dict[str, Any]]  # per-q artifact

    clusters_by_q: List[List[List[int]]]  # per-q clusters over all modes
    cluster_ranges_by_q: List[List[Tuple[float, float]]]  # per-q (fmin,fmax)
    w_dft_by_q: List[ndarray_realFloats]  # per-q (nclusters,)
    Q_cluster_by_q: List[List[ndarray_complex]]  # per-q list of orth bases per cluster

    w_mode_by_q: List[ndarray_realFloats]  # per-q (3N,) GS→ES per-mode weights (DFT-defined)
    valid_modes_by_q: List[np.ndarray]  # per-q indices used in GS→ES score


@dataclass
class ComparisonOutput:
    """Container for DFT cache and ML comparison results.

    Args:
        dft_cache (DFTCache): Cached DFT data.
        results_per_ml (Dict[str, Dict[str, Any]]): Results keyed by ML band path.

    Returns:
        None
    """

    dft_cache: DFTCache
    results_per_ml: Dict[str, Dict[str, Any]]

 
def read_poscar(path: Union[str, Path]) -> Structure:
    """Read a POSCAR/CONTCAR file into a Structure.

    Args:
        path (Union[str, Path]): Path to POSCAR/CONTCAR file.

    Returns:
        Structure: Parsed structure data.
    """

    p = Path(path)
    # Remove blank lines for more robust parsing
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip() != ""]
    if len(lines) < 8:
        raise ValueError(f"Too few lines to be a POSCAR/CONTCAR: {p}")

    # Scale factor and lattice vectors (rows)
    scale = float(lines[1].split()[0])
    if scale <= 0:
        raise ValueError(f"Unsupported scale factor (<=0) in {p}: {scale}")

    lat = np.array([_parse_floats(lines[i], 3) for i in range(2, 5)], dtype=float) * scale  # (3,3)
    idx = 5

    # i had heard that VASP may list either element symbols or counts first; so we detect which.
    tokens = lines[idx].split()
    if _all_int(tokens):
        elements: List[str] = []
        counts = [int(t) for t in tokens]
        idx += 1
    else:
        elements = tokens
        idx += 1
        counts = [int(t) for t in lines[idx].split()]
        idx += 1

    # Total atom count
    n = int(np.sum(counts))
    if n <= 0:
        raise ValueError(f"Invalid atom counts in {p}: {counts}")

    if idx >= len(lines):
        raise ValueError(f"Unexpected EOF in {p} after counts")

    # Optional selective dynamics line
    if lines[idx].split()[0].lower().startswith("s"):
        idx += 1

    if idx >= len(lines):
        raise ValueError(f"Unexpected EOF in {p} before coordinate type")

    # Coordinate type: Direct or Cartesian
    ctype = lines[idx].split()[0].lower()
    idx += 1
    is_direct = ctype.startswith("d")
    is_cart = ctype.startswith("c") or ctype.startswith("k")
    if not (is_direct or is_cart):
        raise ValueError(f"Unknown coordinate type '{lines[idx-1]}' in {p}")

    # Ensure enough coordinate lines remain
    if idx + n > len(lines):
        raise ValueError(f"Not enough coordinate lines in {p}: need {n}, have {len(lines)-idx}")

    coords = np.array([_parse_floats(lines[idx + i], 3) for i in range(n)], dtype=float)  # (N,3)

    # Convert to fractional if coordinates were Cartesian so we can later
    # apply minimum-image wrapping in fractional space.
    if is_direct:
        frac = coords
    else:
        inv_lat = np.linalg.inv(lat)
        frac = coords @ inv_lat  # (N,3) @ (3,3)

    # Return in a normalized structure container (lattice + fractional coords).
    return Structure(lattice=lat, frac=frac, elements=elements, counts=counts)

 
def compute_dq_flat(
    gs: Structure,
    es: Structure,
    masses: ndarray_realFloats,
    lattice_tol: float,
    wrap_minimum_image: bool = True,
    remove_mass_weighted_com: bool = True,
) -> ndarray_realFloats:
    """Compute mass-weighted displacement vector between structures.

    Args:
        gs (Structure): Ground-state structure.
        es (Structure): Excited-state structure.
        masses (ndarray_realFloats): (N,) atomic masses.
        lattice_tol (float): Tolerance for lattice mismatch.
        wrap_minimum_image (bool): Whether to wrap displacements to minimum image.
        remove_mass_weighted_com (bool): Remove mass-weighted center-of-mass shift.

    Returns:
        ndarray_realFloats: (3N,) flattened mass-weighted displacement vector.
    """

    # Sanity checks: same atom count and similar lattice
    if gs.frac.shape != es.frac.shape:
        raise ValueError(f"GS/ES atom count mismatch: {gs.frac.shape} vs {es.frac.shape}")
    if np.max(np.abs(gs.lattice - es.lattice)) > lattice_tol:
        raise ValueError(f"GS/ES lattice mismatch beyond tol={lattice_tol}")

    # Mass array must match number of atoms
    n = int(gs.frac.shape[0])
    masses = np.asarray(masses, dtype=float).reshape(-1)
    if masses.shape != (n,):
        raise ValueError(f"Masses length mismatch: expected {n}, got {masses.shape}")

    # Fractional displacement (ES - GS)
    df = es.frac - gs.frac
    if wrap_minimum_image: # keep displacements relative to the nearest
        # two versions of the atoms, since we are dealing with PBC.
        df = (df + 0.5) % 1.0 - 0.5

    # Convert to Cartesian using GS lattice
    dr = df @ gs.lattice  # (N,3)

    if remove_mass_weighted_com: # needs to be applied before it is mass weighted.
        # basic removal of centre of mass, stops the acoustic phonons from dominating
        w = masses[:, None]
        com = np.sum(w * dr, axis=0) / float(np.sum(masses))
        dr = dr - com[None, :]
    # Apply mass-weighting (equivalent to diagonal mass matrix, but broadcasted)
    dq = (np.sqrt(masses)[:, None] * dr).reshape(3 * n)  # (3N,)
    return dq.astype(float, copy=False)


 
def choose_q_indices(band: BandData, gamma_only: bool, q_tol: float, select_unique_qpts: bool = True) -> List[int]:
    """Select q-point indices to use from band data.
    Robust against duplicate qpoints in DFT band.yaml if gamma_only is true. Now robust against duplicates 
    if gamma_only is not true. we use equality, so if the qpointsa are slightly different (float) then 
    they will not be removed. This can be fixed by supplementing a rounding tool prior. I thought 
    it best to keep to equality for now.
    
    Args:
        band (BandData): Phonon band data.
        gamma_only (bool): If True, select only the q-point closest to Gamma.
        q_tol (float): Tolerance for selecting Gamma.

    Returns:
        List[int]: Indices of selected q-points.
    """

    if gamma_only:
        # Select only the q-point closest to Gamma
        norms = np.linalg.norm(band.q_positions, axis=1)  # (nq,)
        idx = int(np.argmin(norms)) # argmin returns a single index everytime. 
        if float(norms[idx]) > q_tol:
            raise ValueError(
                f"gamma_only=True but nearest q is {band.q_positions[idx].tolist()} (||q||={float(norms[idx])})"
            )
        return [idx]
    
    elif select_unique_qpts: # this ensures there are no qpoint duplicates kept.
        qpts = band.q_positions
        #rounded = np.round(qpts / q_tol).astype(int) 
        _, keep = np.unique(qpts, axis=0, return_index=True)
        # np.unique returns (sorted array, unique indicies)
        keep = np.sort(keep)  # preserve original order
        return keep.tolist()

    # Otherwise use all q-points: indices 0..nq-1
    return list(range(int(band.q_positions.shape[0])))


 
def match_q_indices(dft_qpos: ndarray_realFloats, ml_band: BandData, q_tol: float) -> List[int]:
    """Match DFT q-points to nearest ML q-points.
    This is for robustness, most of the time this will be functionless. 
    In the case of several q points however this is useful. For instance, 
    if for whatever reason the band.yaml contains two identical qpoints (for ML) 
    with data duplication, this returns a singular index, and the remaining qpoint 
    is left alone. 

    This robustness is also implmeneted for DFT.

    Args:
        dft_qpos (ndarray_realFloats): (nq,3) DFT q-point positions.
        ml_band (BandData): ML phonon band data.
        q_tol (float): Maximum distance allowed for a match.

    Returns:
        List[int]: Indices in ml_band matching each DFT q-point.
    """

    # dft_qpos: (nq,3), ml_q: (nq_ml,3)
    ml_q = ml_band.q_positions
    out: List[int] = []
    for q in dft_qpos:
        # Nearest-neighbor match for high symmetry qpoint in reciprocal space.
        d = np.linalg.norm(ml_q - q[None, :], axis=1)  # (nq_ml,)
        j = int(np.argmin(d)) # returns one index. 
        if float(d[j]) > q_tol:
            raise ValueError(
                f"No ML q-point within q_tol={q_tol} of DFT q={q.tolist()} "
                f"(best={ml_q[j].tolist()}, dist={float(d[j])})"
            )
        out.append(j)
    return out

 
def AvgProjPowX_artifact_for_q(
    E_full: ndarray_complex,
    freqs: ndarray_realFloats,
    dq_flat: ndarray_realFloats,
    threshold: float,
    freq_cluster_tol: float,
    top_k_preview: int = 10,
) -> Dict[str, Any]:
    """Compute AvgProjPowX selection artifacts for a single q-point.

    Args:
        E_full (ndarray_complex): (3N,3N) eigenvector matrix.
        freqs (ndarray_realFloats): (3N,) frequencies for the q-point.
        dq_flat (ndarray_realFloats): (3N,) mass-weighted displacement vector (flattened to (3N,) rather than (N,3)
        threshold (float): Cumulative weight threshold for selection (tau)
        freq_cluster_tol (float): Frequency clustering tolerance.
        top_k_preview (int): Number of coupling modes to preview. the details of the first 10 will be displayed.

    Returns:
        Dict[str, Any]: Selection indices, clusters, and summary stats to be printed in the render report
    """

    # Convert to complex to align with complex eigenvectors
    dq = dq_flat.astype(np.complex128, copy=False)
    dq_norm2 = float(np.vdot(dq, dq).real)
    if dq_norm2 <= eps:
        raise ValueError("dq is (near) zero; cannot form projections")

    # Projection per mode: p_m = |e_m^{\dagger} dq|^2 / ||dq||^2
    proj = E_full.conj().T @ dq  # (3N,)
    p = (np.abs(proj) ** 2) / dq_norm2

    # Sort descending and take smallest prefix that reaches threshold
    order = np.argsort(-p, kind="stable")  # indices sorted by descending p. Use
    # a stable sorter  (ends up being mergesort) just because that is resiliant to 
    # mixing p index if p is equivalent. 

    csum = np.cumsum(p[order])  # cumulative contribution
    # np.cumsum is a generator for the sum of X indicies. = Sum top 1, Sum top2 , top 3 ...
    k = int(np.searchsorted(csum, threshold, side="left")) + 1 
    # searchsorted gives you the index of where to put a value to keep an array sorted. 
    # we use it for threshold as it tells us where tau should be put to maintain ordered ness 
    # of csum, and therefore, the index + 1 would have a cumulative sum greater than tau, and index -1
    # would have a cumulative sum less than tau. We want \geq \tau so use +1. 
    # sum top i maps to sum up to and including the ith index in p so these align fine. 
    k = min(k, int(p.size)) # in case np.searchsorted returns len(csum) which would mean
    # we want the len(csum)+1 term in p, but p is only len(csum) long.

    selected = order[:k].tolist()  #couplign modes

    # Group selected modes into near-degenerate frequency clusters
    clusters_selected = cluster_mode_indices_by_frequency(freqs, selected, freq_cluster_tol)

    sum_p = float(np.sum(p))
    warn = None
    if not (0.98 <= sum_p <= 1.02):
        warn = f"sum(p)={sum_p:.6f} outside [0.98,1.02]"

    # Small preview list for logging/inspection
    preview = []
    for idx in order[: min(top_k_preview, int(p.size))]:
        preview.append((int(idx), float(freqs[idx]), float(p[idx])))
    # above will thus show the index, frequency and projection power of the top coupling mdoes
    
    # eigen vectors could be appended but they are of shape (3N,). Can be retrieved from band.yaml with:
    # yam = BandData.from_yaml(path) #use the .eigenvector method
    # evs = yam.eigenvector 
    # print(evs[idx-of-interest])

    return {
        "selected_indices": selected,
        "selected_cumsum_last": float(csum[k - 1]) if k > 0 else 0.0,
        "clusters_selected": clusters_selected,
        "sum_p": sum_p,
        "sum_p_warning": warn,
        "top_contrib_preview": preview,  # (mode, freq, p) small summary only
    }

 
def cluster_mode_indices_by_frequency(
    freqs: ndarray_realFloats,
    indices: Optional[Sequence[int]],
    freq_tol: float,
) -> List[List[int]]:
    """Cluster mode indices by proximity in frequency. This is:
    group by L1 norm \leq freq_tol, otherwise:

    |\omega_{m_{i+1}}(\mathbf q) - \omega_{m_i}(\mathbf q)| > freq_tol → start a new cluster
    
    this gives our eigen spaces.

    Args:
        freqs (ndarray_realFloats): (3N,) frequencies.
        indices (Optional[Sequence[int]]): Mode indices to cluster, or None for all.
        freq_tol (float): Maximum frequency gap within a cluster.

    Returns:
        List[List[int]]: Clusters of mode indices.
    """

    # Use all indices if none provided
    # Gets an array of the indices by the arange generator or by what is specified
    # (the coupling indicies)

    if indices is None:
        idx = np.arange(freqs.size, dtype=int)
    else:
        idx = np.asarray(list(indices), dtype=int)

    # Sort selected indices by frequency (ascending)
    f = freqs[idx]
    order = np.argsort(f, kind="stable") # again we use stable to ensure that index is 
    # conserved if two members of the array are equivalent.
    idx_sorted = idx[order].tolist() # indicies of where frequencies appear such that 
    # they are in ascending order. idx_sorted[0] lowest frequency index etc... 
    f_sorted = f[order] # the sorted frequencies themselves. 

    clusters: List[List[int]] = []
    cur: List[int] = [idx_sorted[0]]  # initialise our cluster. cur holds 
    # the indices of the clustered modes. 
    for i in range(1, len(idx_sorted)):
        # Start new cluster when frequency gap exceeds tolerance
        if float(f_sorted[i] - f_sorted[i - 1]) > freq_tol:
            clusters.append(cur) # stash cur
            cur = [idx_sorted[i]] # reset cur (the beginning of a new cluster.)
        else:
            cur.append(idx_sorted[i])
    clusters.append(cur)
    return clusters

 
def orth(A: ndarray_complex, rcond: float = eps) -> ndarray_complex:
    """Compute an orthonormal basis for the column space of A.

    Args:
        A (ndarray_complex): Input matrix.
        rcond (float): Relative tolerance for rank determination.

    Returns:
        ndarray_complex: Orthonormal basis matrix with shape (m, r).
    """

    # QR-based orthonormal basis with rank cutoff
    if A.size == 0:
        return A[:, :0].copy()
    Q, R = np.linalg.qr(A, mode="reduced")  # Q: (m,k), R: (k,n), k=min(m,n)
    # "reduced" gives just the orthonormal columns spanning col(A) (no extra basis vectors)
    if R.size == 0:
        return Q[:, :0].copy()
    d = np.abs(np.diag(R))
    if d.size == 0:
        return Q[:, :0].copy()
    # Keep columns with diagonal(R) above tolerance. We want to get rid of any nearly 
    # dependent columns, by nearly we mean vanishingly small, rcond is 1e-12 (eps). 
    tol = float(d.max()) * float(rcond)
    r = int(np.sum(d > tol))
    return Q[:, :r]

 
def principal_angles(Qd: ndarray_complex, Qm: ndarray_complex) -> Tuple[ndarray_realFloats, ndarray_realFloats, float]:
    """Compute principal angles between two subspaces.

    Args:
        Qd (ndarray_complex): Orthonormal basis for DFT subspace.
        Qm (ndarray_complex): Orthonormal basis for ML subspace.

    Returns:
        Tuple[ndarray_realFloats, ndarray_realFloats, float]: Singular values, angles (deg), and X score. 
        (avg projection power)
    """

    # Qd: (3N,kd), Qm: (3N,km)
    kd = int(Qd.shape[1])
    if kd == 0: # these cases need to be handled as it is possible that there are 
        # no modes in the frequency of the coupling mode under study. 
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), float("nan")
    km = int(Qm.shape[1])
    if km == 0:
        return np.zeros((0,), dtype=float), np.full((0,), 90.0, dtype=float), 0.0

    # Overlap matrix and its singular values define principal angles
    C = Qd.conj().T @ Qm  # (kd,km)
    sig = np.linalg.svd(C, compute_uv=False)
    sig = np.clip(sig.real.astype(float, copy=False), 0.0, 1.0) # should already be in [0,1] 
    # but we are dealing with floats... 
    theta = np.degrees(np.arccos(sig))
    X = float(np.sum(sig**2) / kd)
    return sig, theta, X

 
def acoustic_mode_indices(
    freqs: ndarray_realFloats, n_acoustic: int = 3, abs_tol: float = 1e-4
) -> np.ndarray:
    """Return indices of acoustic translational modes (Gamma) robustly. 
    (just takes the top 3 coupling modes. this is irrelevant if the center of mass 
    displacement has been removed from the displacement vector (default is that it has).)"""
    f = np.asarray(freqs, dtype=float)  # (3N,)
    af = np.abs(f)
    near0 = np.where(af <= abs_tol)[0]
    if near0.size >= n_acoustic:
        order = np.argsort(af[near0], kind="stable")[:n_acoustic]
        return near0[order]
    return np.argsort(af, kind="stable")[:n_acoustic]


 
def dft_mode_weights_for_q(
    E_full: ndarray_complex,
    freqs: ndarray_realFloats,
    dq_flat: ndarray_realFloats,
    *,
    kind: str = "p",  # "p", "S", "lambda"
    n_acoustic: int = 3,
    acoustic_abs_tol: float = 1e-4,
) -> Tuple[ndarray_realFloats, np.ndarray]:
    
    """
    DFT-defined per-mode weights for DeltaQ scores and MLIP ranking.

    This builds a per-mode importance weight using the DFT eigenvectors and
    the GS to ES mass-weighted displacement 'dq'. These weights are then used to
    compute weighted RMS errors (frequency + eigenvector mismatch) when matching
    ML modes to DFT modes via the assignment step. In other words, this function
    decides which DFT modes "matter" most for the ranking of MLIPs.
    
    a measures projection of weighted displacement vector onto eigenmode.

    The returned 'w' and 'valid' are consumed in 'gses_score_from_assignment':
      - 'w' supplies the weights in the RMS metrics (E_freq, E_vec, Score).
      - 'valid' selects the subset of modes that actually contribute (w > 0).

    use to rank mlips by a) p => how well the mode describes the geometry change 
    b) S or Lambda, they decrease weight of soft modes and increase the 
    weight of higher frequency modes -- typically use lambda for this because in the harmonid 
    approx energy goes with w^2 a^2. 
     
     so b) used to answer: does the mlip reproduce the energetically 
    important eignemodes? 

    use a) to answer: does the mlip reprorduce the geometrically importat eigenodes? 


    kind:
      - "p": |a|^2
      - "S":  \omega|a|^2
      - "lambda": \omega^2|a|^2
    """
    # Project the displacement onto eigenvectors to get per-mode amplitudes.
    # These amplitudes quantify how strongly each DFT mode participates
    # in the structural change (GS→ES) encoded in dq.
    dq = dq_flat.astype(np.complex128, copy=False)
    proj = E_full.conj().T @ dq # (3N,) 
    a2 = (np.abs(proj) ** 2).astype(float) # |a|^2
    # Frequency magnitudes (non-negative), used to scale weights by energy.
    omega = np.abs(np.asarray(freqs, dtype=float))  # |\omega|

    if kind == "p":
        # Pure projection power: importance is just how much dq lies in each mode.
        w = a2
    elif kind == "S":
        # weight by |\omega| to emphasize higher-frequency contributions mildly.
        w = omega * a2
    elif kind == "lambda":
        # Weight by |\omega|^2 to emphasize higher-frequency contributions  strongly.
        w = (omega ** 2) * a2
    else:
        raise ValueError(f"Unknown weight kind='{kind}' (use 'p','S','lambda')")

    # Remove acoustic translations at Gamma (zero their weights)
    # commented out because of the implemented COM removal.
    # ac = acoustic_mode_indices(freqs, n_acoustic=n_acoustic, abs_tol=acoustic_abs_tol)
    # w[ac] = 0.0

    # Only modes with nonzero weight are considered "important" for the GS to ES score (called dQ score).
    # This avoids polluting the weighted RMS with modes that dq does not excite.
    valid = np.where(w > 0.0)[0]  # indices with nonzero weight
    return w, valid


 
def overlap_sq(E_dft: ndarray_complex, E_ml: ndarray_complex) -> ndarray_realFloats:
    """Compute phase-invariant overlaps between DFT and ML eigenvector bases.

    Feed this into the Hungarian assignment in the GS->ES dQ score. We maximize
    the diagonal overlap between DFT and ML modes to get a one-to-one
    matching before scoring frequency/eigenvector errors.

    Shapes:
      - E_dft:(3N,3N) DFT eigenvectors (columns are modes).
      - E_ml: (3N,3N) ML eigenvectors (columns are modes).
      - O: (3N,3N) overlaps in [0,1], O_ij = |e_i^H e_j|^2.

    gives the cost matrix for hungarian_min. 
    """
    # Overlap between DFT and ML eigenvector bases (phase invariant).
    C = E_dft.conj().T @ E_ml  # (3N,3N)
    return (np.abs(C) ** 2).astype(float, copy=False)  # elementwise |.|^2

 
def hungarian_min(cost: ndarray_realFloats) -> np.ndarray:
    """
    Solve min-cost assignment for a square cost matrix (n,n).
    Returns assignment a where a[i] = j chosen for row i.

    hungarian minimisatino algo.
    used by `hungarian_maximize`, which is used to match DFT modes to ML modes 
    for the GS->ES dQ score.

    cost: (n,n) array.
    return: (n,) int array mapping row i to chosen column a[i].
    """
    a = np.asarray(cost, dtype=float)  # (n,n) cost matrix
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"hungarian_min requires square matrix, got {a.shape}")
    n = int(a.shape[0])

    # Use 1-based indexing internally to match the classic Hungarian pseudocode.
    u = np.zeros(n + 1, dtype=float) # row potentials
    v = np.zeros(n + 1, dtype=float) # column potentials
    p = np.zeros(n + 1, dtype=int) # matching for columns: p[j] = i
    way = np.zeros(n + 1, dtype=int) # predecessor tracking

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=float)
        used = np.zeros(n + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assign = np.empty(n, dtype=int)  # row to column assignment
    for j in range(1, n + 1):
        i = p[j]
        if i != 0:
            assign[i - 1] = j - 1
    return assign

 
def hungarian_maximize(weight: ndarray_realFloats) -> np.ndarray:
    """Maximize sum weight[i, assign[i]] for square matrix.

    Shapes:
      - weight: (n,n) overlap/score matrix.
      - return: (n,) assignment array.
    """
    W = np.asarray(weight, dtype=float)  # (n,n) weight/overlap matrix
    wmax = float(np.max(W)) if W.size else 0.0
    cost = wmax - W
    return hungarian_min(cost)

 
def gses_score_from_assignment(
    freqs_dft: ndarray_realFloats,
    freqs_ml: ndarray_realFloats,
    w_mode: ndarray_realFloats,
    valid_modes: np.ndarray,
    assign: np.ndarray,
    O: ndarray_realFloats,
    *,
    omega_floor: float = 1e-4,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Compute dQ score from a DFT→ML assignment:
      - E_freq (weighted RMS abs error)
      - E_freq_rel (weighted RMS relative error with omega_floor)
      - E_vec (weighted RMS eigenvector mismatch)
      - Score = E_freq + alpha * E_vec

    This is the GS->ES score ranking channel. It is called from 'compare_one_ml'
    after the optimal DFT->ML assignment is found, and it uses DFT-defined
    per-mode weights (from 'dft_mode_weights_for_q') to emphasize the
    displacement-relevant modes.

    Shapes:
    freqs_dft, freqs_ml: (3N,) arrays.
    w_mode: (3N,) weights.
    valid_modes: (n_valid,) indices where w_mode > 0.
    assign: (3N,) DFT->ML mapping.
    O: (3N,3N) overlap matrix.
    """
    # Shapes: freqs_* are (3N,), w_mode is (3N,), idx is list of valid mode indices.
    f_d = np.asarray(freqs_dft, dtype=float)
    f_m = np.asarray(freqs_ml, dtype=float)
    w = np.asarray(w_mode, dtype=float)
    idx = np.asarray(valid_modes, dtype=int)

    if idx.size == 0:
        return {
            "E_freq": float("nan"),
            "E_freq_rel": float("nan"),
            "E_vec": float("nan"),
            "Score": float("nan"),
        }

    # Assignment: pi[i] is the ML mode index matched to DFT mode i.
    pi = np.asarray(assign, dtype=int)  # length 3N, maps DFT -> ML
    j = pi[idx]  # ML mode indices matched to each DFT mode in idx

    # Frequency errors on valid modes (matched DFT->ML pairs).
    dw = f_m[j] - f_d[idx]
    denom = float(np.sum(w[idx]))
    if denom <= eps:
        # Fallback to uniform weights on idx if weights vanish.
        wf = np.ones(idx.size, dtype=float)
        denom = float(np.sum(wf))
    else:
        wf = w[idx]

    E_freq = float(np.sqrt(np.sum(wf * (dw ** 2)) / denom))

    # Relative errors with floor to avoid div-by-zero near acoustic modes
    rel = dw / (np.abs(f_d[idx]) + float(omega_floor))
    E_freq_rel = float(np.sqrt(np.sum(wf * (rel ** 2)) / denom))

    # Overlaps for matched modes (values in [0,1]) quantify eigenvector similarity.
    odiag = O[idx, j]
    E_vec = float(np.sqrt(np.sum(wf * (1.0 - odiag)) / denom))

    Score = float(E_freq + float(alpha) * E_vec)

    return {
        "E_freq": E_freq,
        "E_freq_rel": E_freq_rel,
        "E_vec": E_vec,
        "Score": Score,
    }

 
def cluster_basis_and_weight(
    E_full: ndarray_complex,
    dq: ndarray_complex,
    dq_norm2: float,
    cluster: Sequence[int],
) -> Tuple[ndarray_complex, float]:
    """Compute cluster basis and its weight on dq.
    Used in `build_dft_cache` to convert each DFT frequency cluster into
    a subspace basis and a scalar coupling weight. These are later used
    to identify relevant clusters 

    Shapes:
    E_full: (3N,3N) eigenvector matrix.
    cluster: list of mode indices, length k_cluster.
    Q: (3N,k_cluster) orthonormal basis.
    w: scalar in [0,1], fraction of dq captured by this cluster.

    Args:
        E_full (ndarray_complex): (3N,3N) eigenvector matrix.
        dq (ndarray_complex): (3N,) mass-weighted displacement vector.
        dq_norm2 (float): Squared norm of dq.
        cluster (Sequence[int]): Mode indices in the cluster.

    Returns:
        Tuple[ndarray_complex, float]: Orthonormal basis Q and weight w.
    """

    # Build cluster subspace from selected eigenvectors.
    A = E_full[:, list(cluster)]  # (3N, k_cluster)
    Q = orth(A)
    if Q.shape[1] == 0:
        return Q, 0.0
    # Weight = projection of dq onto cluster subspace.
    v = Q.conj().T @ dq
    w = float(np.sum(np.abs(v) ** 2) / dq_norm2)
    return Q, w

  
def top_clusters_by_weight(weights: ndarray_realFloats, threshold: float) -> List[int]:
    """Select cluster indices by cumulative weight threshold. (\tau)

    Args:
        weights (ndarray_realFloats): Cluster weights.
        threshold (float): Cumulative weight threshold.

    Returns:
        List[int]: Indices of selected clusters.
    """

    # select highest-weight clusters until cumulative threshold reached
    # same as what was done to the modes. 
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return []
    order = np.argsort(-w, kind="stable")
    c = np.cumsum(w[order])
    k = int(np.searchsorted(c, threshold, side="left")) + 1
    k = min(k, int(w.size))
    return order[:k].tolist()

 
def build_dft_cache(
    contcar_gs: Union[str, Path],
    contcar_es: Union[str, Path],
    band_dft_path: Union[str, Path],
    q_tol: float,
    lattice_tol: float,
    threshold: float,
    freq_cluster_tol: float,
    freq_window: float,
    remove_mass_weighted_com: bool,
    gamma_only: bool,
    weight_type: str = "p", # "p", "S", or "lambda"
) -> DFTCache:
    """Build cached DFT data and derived quantities.

    One-time preprocessing step. Computes the GS->ES displacement (dq),
    loads DFT phonons, selects q-points, and derives all DFT-side        
    quantities needed to compare *multiple* MLIPs efficiently.
    The resulting cache contains per-q eigenvectors, per-mode weights,
    frequency clusters, and cluster subspace weights used in ranking.

    Shapes (key cached fields):
      dq_flat: (3N,) mass-weighted displacement.
     freqs_by_q: list of (3N,) arrays.
      E_by_q: list of (3N,3N) eigenvector matrices.
      clusters_by_q: list of list-of-mode-index clusters.
      Q_cluster_by_q: list of list of (3N,k_cluster) bases.
      w_dft_by_q: list of (ncluster,) DFT cluster weights.

    Args:
        contcar_gs (Union[str, Path]): Ground-state CONTCAR path.
        contcar_es (Union[str, Path]): Excited-state CONTCAR path.
        band_dft_path (Union[str, Path]): DFT band.yaml path.
        q_tol (float): Q-point match tolerance.
        lattice_tol (float): Lattice mismatch tolerance.
        threshold (float): Cumulative weight threshold.
        freq_cluster_tol (float): Frequency cluster tolerance.
        freq_window (float): Frequency window for ML clustering.
        remove_mass_weighted_com (bool): Remove COM shift if True.
        gamma_only (bool): Only use Gamma if True.

    Returns:
        DFTCache: Cached DFT-derived data.
    """

    # Load ground/excited structures (same atom ordering and similar lattice).
    gs = read_poscar(contcar_gs)
    es = read_poscar(contcar_es)

    # Load DFT band data (phonopy band.yaml).
    dft = BandData.from_yaml(band_dft_path)

    if dft.natom != int(gs.frac.shape[0]):
        raise ValueError(f"DFT natom ({dft.natom}) != CONTCAR natom ({int(gs.frac.shape[0])})")

    # Mass-weighted displacement vector between GS and ES.
    dq_flat = compute_dq_flat(
        gs=gs,
        es=es,
        masses=dft.masses,
        lattice_tol=lattice_tol,
        wrap_minimum_image=True,
        remove_mass_weighted_com=remove_mass_weighted_com,
    )
    # Complex dtype for projections with complex eigenvectors.
    dq = dq_flat.astype(np.complex128, copy=False)
    dq_norm2 = float(np.vdot(dq, dq).real)
    if dq_norm2 <= eps:
        raise ValueError("dq is (near) zero; check CONTCAR_gs/CONTCAR_es consistency")

    # Select which q-points to use (Gamma-only or all), and de-duplicate if needed.
    q_indices = choose_q_indices(dft, gamma_only=gamma_only, q_tol=q_tol)
    q_positions = dft.q_positions[q_indices].copy()

    # Per-q storage for frequencies, eigenvectors, and selection artifacts.
    freqs_by_q: List[ndarray_realFloats] = [] # ndarray_realFloats for arryas expected to be floats
    E_by_q: List[ndarray_complex] = [] #ndarray_complex for arrays expected to be complex
    AvgProjPowX_by_q: List[Dict[str, Any]] = []

    # Per-q frequency clusters and their subspace information (used in research metrics).
    clusters_by_q: List[List[List[int]]] = []
    cluster_ranges_by_q: List[List[Tuple[float, float]]] = []
    w_dft_by_q: List[ndarray_realFloats] = []
    Q_cluster_by_q: List[List[ndarray_complex]] = []

    w_mode_by_q: List[ndarray_realFloats] = []
    valid_modes_by_q: List[np.ndarray] = []

    for qi in q_indices:
        # Pull frequencies and eigenvectors for each selected q-point.
        freqs = dft.frequencies[qi].astype(float, copy=False)
        if freqs.shape[0] != 3 * dft.natom:
            raise ValueError(f"DFT nmodes mismatch at q_idx={qi}: got {freqs.shape[0]}")
        E_full = dft.E(qi, normalize=True)

        # --- GS→ES per-mode weights (DFT-defined) ---
        # These weights determine which modes matter most for the dQ score
        # used in the final MLIP ranking.
        w_mode, valid_modes = dft_mode_weights_for_q(
            E_full=E_full,
            freqs=freqs,
            dq_flat=dq_flat,
            kind=weight_type,
            n_acoustic=3,
            acoustic_abs_tol=1e-4,
        )

        # AvgProjPowX is the average projected power of the mlip
        # coupling modes onto DFT's coupling modes. X is just kept
        # because that is what i initially called it mathematically haha ...
        AvgProjPowX = AvgProjPowX_artifact_for_q(
            E_full=E_full,
            freqs=freqs,
            dq_flat=dq_flat,
            threshold=threshold,
            freq_cluster_tol=freq_cluster_tol,
        )

        # Cluster all modes by frequency proximity (degeneracy groups).
        clusters = cluster_mode_indices_by_frequency(freqs, indices=None, freq_tol=freq_cluster_tol)

        Qs: List[ndarray_complex] = []
        ws: List[float] = []
        ranges: List[Tuple[float, float]] = []
        for cl in clusters:
            # Cluster frequency range and its projection weight.
            fcl = freqs[list(cl)]
            fmin = float(np.min(fcl))
            fmax = float(np.max(fcl))
            Qc, wc = cluster_basis_and_weight(E_full, dq, dq_norm2, cl)
            Qs.append(Qc)
            ws.append(wc)
            ranges.append((fmin, fmax))

        freqs_by_q.append(freqs.copy())
        E_by_q.append(E_full)
        AvgProjPowX_by_q.append(AvgProjPowX)

        # Cache DFT cluster data for later ML comparisons.
        clusters_by_q.append(clusters)
        cluster_ranges_by_q.append(ranges)
        w_dft_by_q.append(np.asarray(ws, dtype=float))
        Q_cluster_by_q.append(Qs)

        # Cache the per-mode weights and valid indices for the dQ score.
        w_mode_by_q.append(w_mode.copy())  # (3N,)
        valid_modes_by_q.append(valid_modes.copy())  # (n_valid,)

    # DFTCache is reused for every ML model (avoids recomputing DFT-side quantities).
    return DFTCache(
        dq_flat=dq_flat,
        dq_norm2=dq_norm2,
        masses=dft.masses.copy(),
        dft_path=str(Path(band_dft_path)),
        q_indices=q_indices,
        q_positions=q_positions,
        freqs_by_q=freqs_by_q,
        E_by_q=E_by_q,
        AvgProjPowX_by_q=AvgProjPowX_by_q,
        clusters_by_q=clusters_by_q,
        cluster_ranges_by_q=cluster_ranges_by_q,
        w_dft_by_q=w_dft_by_q,
        Q_cluster_by_q=Q_cluster_by_q,
        w_mode_by_q=w_mode_by_q,
        valid_modes_by_q=valid_modes_by_q,
    )

   
def compare_one_ml(
    cache: DFTCache,
    ml_path: Union[str, Path],
    q_tol: float,
    threshold: float,
    freq_cluster_tol: float,
    freq_window: float,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """Compare one ML band.yaml against cached DFT data.
        Core per-model evaluation. Computes: AvgProjPowX subspace agreement, cluster-window
        agreement, and GS->ES dQ score used for final ranking.

    Shapes:
      cache.E_by_q[iq]: (3N,3N) DFT eigenvector matrix.
      freqs_ml: (3N,) ML frequencies at matched q.
      AvgProjPowX_per_q: list of dicts with X, sigma, theta, etc.
      research_per_q: list of dicts with cluster-window metrics.
      gses_per_q: list of dicts with E_freq/E_vec/Score.

    Args:
        cache (DFTCache): Cached DFT data.
        ml_path (Union[str, Path]): ML band.yaml path.
        q_tol (float): Q-point match tolerance.
        threshold (float): Cumulative weight threshold.
        freq_cluster_tol (float): Frequency cluster tolerance.
        freq_window (float): Frequency window for ML clustering.

    Returns:
        Dict[str, Any]: Comparison results for AvgProjPowX and coupling cluster analysis.
    """
    ml = BandData.from_yaml(ml_path)

    n = int(cache.masses.shape[0])
    if ml.natom != n:
        raise ValueError(f"ML natom ({ml.natom}) != DFT natom ({n}) for {ml_path}")

    # match DFT q-points to nearest ML q-points.
    ml_q_indices = match_q_indices(cache.q_positions, ml, q_tol=q_tol)

    # reuse cached dq for projections.
    dq = cache.dq_flat.astype(np.complex128, copy=False)
    dq_norm2 = cache.dq_norm2

    # (AvgProjPowX) per-q and aggregated.
    AvgProjPowX_per_q: List[Dict[str, Any]] = []
    Xs: List[float] = []

    research_per_q: List[Dict[str, Any]] = []
    l1s: List[float] = []
    angle_scores: List[float] = []
    sigma2_scores: List[float] = []

    # GS->ES targeted (dQ score) metrics per-qm=, aggregated
    gses_per_q: List[Dict[str, Any]] = []
    gses_scores: List[float] = []
    gses_Efreq: List[float] = []
    gses_Evec: List[float] = []
    gses_EfreqRel: List[float] = []

    for iq, (dft_qi, ml_qi) in enumerate(zip(cache.q_indices, ml_q_indices)):
        # Pull ML data for matched q-point.
        freqs_ml = ml.frequencies[ml_qi].astype(float, copy=False)
        if freqs_ml.shape[0] != 3 * ml.natom:
            raise ValueError(f"ML nmodes mismatch at q_idx={ml_qi} for {ml_path}")

        E_ml = ml.E(ml_qi, normalize=True)  # (3N,3N)

        # --- GS→ES targeted: DFT-weighted optimal mode matching + score ---
        E_dft = cache.E_by_q[iq]  # (3N,3N)
        O = overlap_sq(E_dft, E_ml)  # (3N,3N) overlaps
        assign = hungarian_maximize(O)  # length 3N, DFT->ML assignment

        w_mode = cache.w_mode_by_q[iq]  # (3N,)
        valid_modes = cache.valid_modes_by_q[iq]  # subset of indices

        metrics = gses_score_from_assignment(
            freqs_dft=cache.freqs_by_q[iq],
            freqs_ml=freqs_ml,
            w_mode=w_mode,
            valid_modes=valid_modes,
            assign=assign,
            O=O,
            omega_floor=1e-4,
            alpha=alpha,
        )

        gses_per_q.append(
            {
                "dft_q_index": int(dft_qi),
                "ml_q_index": int(ml_qi),
                "q_position": cache.q_positions[iq].astype(float),
                **metrics,
            }
        )
        gses_scores.append(float(metrics["Score"]))
        gses_Efreq.append(float(metrics["E_freq"]))
        gses_EfreqRel.append(float(metrics["E_freq_rel"]))
        gses_Evec.append(float(metrics["E_vec"]))

        # Compute AvgProjPowX selection on ML side.
        AvgProjPowX_ml = AvgProjPowX_artifact_for_q(
            E_full=E_ml,
            freqs=freqs_ml,
            dq_flat=cache.dq_flat,
            threshold=threshold,
            freq_cluster_tol=freq_cluster_tol,
        )

        # Compare subspaces spanned by selected modes.
        dft_AvgProjPowX = cache.AvgProjPowX_by_q[iq]
        dft_sel = dft_AvgProjPowX["selected_indices"]
        ml_sel = AvgProjPowX_ml["selected_indices"]

        # Build orthonormal bases for the selected coupling modes.
        Qd = orth(cache.E_by_q[iq][:, dft_sel])  # (3N,k_dft)
        Qm = orth(E_ml[:, ml_sel])  # (3N,k_ml)
        sig, theta, X = principal_angles(Qd, Qm)

        AvgProjPowX_per_q.append(
            {
                "dft_q_index": int(dft_qi),
                "ml_q_index": int(ml_qi),
                "q_position": cache.q_positions[iq].astype(float),
                "k_dft": int(Qd.shape[1]),
                "k_ml": int(Qm.shape[1]),
                "sigma": sig,
                "theta_deg": theta,
                "X": float(X),
                "dft_sum_p_warning": dft_AvgProjPowX.get("sum_p_warning", None),
                "ml_sum_p_warning": AvgProjPowX_ml.get("sum_p_warning", None),
            }
        )
        Xs.append(float(X))

        # Pull data from DFT cache (so it is not computed every time).
        clusters = cache.clusters_by_q[iq]
        ranges = cache.cluster_ranges_by_q[iq]
        w_dft = cache.w_dft_by_q[iq]
        Qd_clusters = cache.Q_cluster_by_q[iq]

        # obtain the "coupling clusters" (highest DFT-weighted clusters).
        relevant = top_clusters_by_weight(w_dft, threshold=threshold)

        cluster_rows: List[Dict[str, Any]] = []
        w_ml_list: List[float] = []
        mean_theta_list: List[float] = []
        mean_sigma2_list: List[float] = []

        for cid, (cl, (fmin, fmax), QdC, wd) in enumerate(zip(clusters, ranges, Qd_clusters, w_dft)):
            # Window ML frequencies around each DFT cluster. These are the frequency bins to form
            # ml clusters. Aim is to test whether ML captures the the same modes in teh same frequency domains.
            lo = fmin - freq_window
            hi = fmax + freq_window
            window_idx = np.where((freqs_ml >= lo) & (freqs_ml <= hi))[0].astype(int).tolist()
            # window_idx gives the indices for where the ML freqs are in the respetive bin
            # (we loop over the dft clusters)

            if window_idx:
                # Compare cluster subspaces and weights within the window
                QmW = orth(E_ml[:, window_idx])
                v = QmW.conj().T @ dq
                wml = float(np.sum(np.abs(v) ** 2) / dq_norm2)
                sigC, thetaC, _ = principal_angles(QdC, QmW)
                mean_theta = float(np.mean(thetaC)) if thetaC.size else 90.0
                max_theta = float(np.max(thetaC)) if thetaC.size else 90.0
                sigma2 = float(np.sum(sigC**2) / max(int(QdC.shape[1]), 1)) if sigC.size else 0.0
            else:
                wml = 0.0
                sigC = np.zeros((0,), dtype=float)
                thetaC = np.zeros((0,), dtype=float)
                mean_theta = 90.0
                max_theta = 90.0
                sigma2 = 0.0

            w_ml_list.append(wml)
            mean_theta_list.append(mean_theta)
            mean_sigma2_list.append(sigma2)

            if cid in set(relevant):
                cluster_rows.append(
                    {
                        "cluster_id": int(cid),
                        "size_dft": int(len(cl)),
                        "freq_range_dft": (float(fmin), float(fmax)),
                        "w_dft": float(wd),
                        "w_ml_window": float(wml),
                        "theta_mean_deg": float(mean_theta),
                        "theta_max_deg": float(max_theta),
                        "sigma": sigC,
                    }
                )

        # Aggregate per-q measurement  for relevant clusters (coupling clusters)
        w_ml = np.asarray(w_ml_list, dtype=float)
        l1 = float(np.sum(np.abs(w_ml[relevant] - w_dft[relevant]))) if relevant else 0.0
        angle_score = float(np.sum(w_dft[relevant] * np.asarray(mean_theta_list, dtype=float)[relevant])) if relevant else 0.0
        sigma2_score = float(
            np.sum(w_dft[relevant] * (1.0 - np.asarray(mean_sigma2_list, dtype=float)[relevant]))
        ) if relevant else 0.0

        research_per_q.append(
            {
                "dft_q_index": int(dft_qi),
                "ml_q_index": int(ml_qi),
                "q_position": cache.q_positions[iq].astype(float),
                "relevant_clusters": relevant,
                "clusters_relevant": cluster_rows,
                "summary": {
                    "L1_weights_relevant": l1,
                    "weighted_mean_theta_deg_relevant": angle_score,
                    "weighted_1_minus_sigma2_relevant": sigma2_score,
                },
            }
        )
        l1s.append(l1)
        angle_scores.append(angle_score)
        sigma2_scores.append(sigma2_score)

    # Summaries across q-points (per-MLIP aggregates used in the final ranking).
    X_arr = np.asarray(Xs, dtype=float)
    AvgProjPowX_summary = {
        "X_mean": float(np.mean(X_arr)) if X_arr.size else float("nan"),
        "X_min": float(np.min(X_arr)) if X_arr.size else float("nan"),
        "X_max": float(np.max(X_arr)) if X_arr.size else float("nan"),
        "n_q": int(len(Xs)),
    }

    research_summary = {
        "L1_weights_mean": float(np.mean(l1s)) if l1s else float("nan"),
        "weighted_mean_theta_deg_mean": float(np.mean(angle_scores)) if angle_scores else float("nan"),
        "weighted_1_minus_sigma2_mean": float(np.mean(sigma2_scores)) if sigma2_scores else float("nan"),
        "n_q": int(len(l1s)),
    }

    gses_summary = {
        "Score_mean": float(np.mean(gses_scores)) if gses_scores else float("nan"),
        "Score_min": float(np.min(gses_scores)) if gses_scores else float("nan"),
        "E_freq_mean": float(np.mean(gses_Efreq)) if gses_Efreq else float("nan"),
        "E_freq_rel_mean": float(np.mean(gses_EfreqRel)) if gses_EfreqRel else float("nan"),
        "E_vec_mean": float(np.mean(gses_Evec)) if gses_Evec else float("nan"),
        "n_q": int(len(gses_scores)),
    }

    return {
        "AvgProjPowX": {"per_q": AvgProjPowX_per_q, "summary": AvgProjPowX_summary},
        "research": {"per_q": research_per_q, "summary": research_summary},
        "gses": {"per_q": gses_per_q, "summary": gses_summary},
    }

 
def run(
    contcar_gs: str,
    contcar_es: str,
    band_dft_path: str,
    band_ml_paths: List[str],
    q_tol: float,
    lattice_tol: float,
    threshold: float,
    freq_cluster_tol: float,
    freq_window: float,
    remove_mass_weighted_com: bool,
    gamma_only: bool,
    alpha: float, 
    weight_kind: str,
) -> ComparisonOutput:
    """Run DFT/ML band comparison and return results. build DFT cache once, then evaluate
        each MLIP band.yaml against it. This is the programmatic entry point
        used by the CLI and downstream scripts.

    Args:
        contcar_gs (str): Ground-state CONTCAR path.
        contcar_es (str): Excited-state CONTCAR path.
        band_dft_path (str): DFT band.yaml path.
        band_ml_paths (List[str]): ML band.yaml paths.
        q_tol (float): Q-point match tolerance.
        lattice_tol (float): Lattice mismatch tolerance.
        threshold (float): Cumulative weight threshold.
        freq_cluster_tol (float): Frequency cluster tolerance.
        freq_window (float): Frequency window for ML clustering.
        remove_mass_weighted_com (bool): Remove COM shift if True.
        gamma_only (bool): Only use Gamma if True.

    Returns:
        ComparisonOutput: Cached DFT data and ML comparison results.
    """

    # Build DFT cache once, then compare against each ML model
    cache = build_dft_cache(
        contcar_gs=contcar_gs,
        contcar_es=contcar_es,
        band_dft_path=band_dft_path,
        q_tol=q_tol,
        lattice_tol=lattice_tol,
        threshold=threshold,
        freq_cluster_tol=freq_cluster_tol,
        freq_window=freq_window,
        remove_mass_weighted_com=remove_mass_weighted_com,
        gamma_only=gamma_only,
        weight_type=weight_kind,
    )

    results: Dict[str, Dict[str, Any]] = {}
    for mlp in band_ml_paths:
        # Compute comparison for each ML band.yaml
        results[str(mlp)] = compare_one_ml(
            cache=cache,
            ml_path=mlp,
            q_tol=q_tol,
            threshold=threshold,
            freq_cluster_tol=freq_cluster_tol,
            freq_window=freq_window,
            alpha = alpha
        )

    return ComparisonOutput(dft_cache=cache, results_per_ml=results)

 
def render_report(
    out: ComparisonOutput,
    threshold: float,
    freq_cluster_tol: float,
    freq_window: float,
    *,
    alpha: float,
    weight_kind: str,
) -> str:
    """create comparison report.
    """

    # Render a plain-text report with per-q summaries
    c = out.dft_cache
    lines: List[str] = []

    def h(s: str) -> None:
        # Section header helper
        lines.append(s)
        lines.append("-" * len(s))

    def fmt_q(q: ndarray_realFloats) -> str:
        # Consistent q-position formatting
        return f"[{q[0]: .6f}, {q[1]: .6f}, {q[2]: .6f}]"

    # High-level report header uses dQ 
    h("Mode–coupling comparison report")
    lines.append(f"DFT: {c.dft_path}")
    lines.append(f"N atoms: {int(c.masses.shape[0])}    N modes: {int(c.dq_flat.size)}")
    # Settings printed here should include the knobs that affect final ranking.
    lines.append(
        "Settings: "
        f"threshold={threshold:.3f}, "
        f"freq_cluster_tol={freq_cluster_tol:.3f}, "
        f"freq_window={freq_window:.3f}, "
        f"alpha={float(alpha):.3f}, "
        f"weight_kind={str(weight_kind)}"
    )
    lines.append(f"q-points used: {len(c.q_indices)}")
    for i, qi in enumerate(c.q_indices):
        lines.append(f"  q[{i}] index={qi}  pos={fmt_q(c.q_positions[i])}")
    lines.append("")

    # DFT-side sanity preview of projection weights
    h("DFT Coupling modes  (per mode) ")
    for i, qi in enumerate(c.q_indices):
        leg = c.AvgProjPowX_by_q[i]
        warn = leg.get("sum_p_warning", None)
        lines.append(f"q[{i}] index={qi}  sum(p)={leg['sum_p']:.6f}" + (f"  WARNING: {warn}" if warn else ""))
        lines.append(f"  selected modes: k={len(leg['selected_indices'])}  cumsum_last={leg['selected_cumsum_last']:.6f}")
        lines.append("  Coupling modes: mode | freq | p):")
        for (m, f, p) in leg["top_contrib_preview"]:
            lines.append(f"    {m:4d}  {f: .6f}  {p: .6e}")
        lines.append("")

    # DFT clusters ranked by their weights in dq
    h("DFT Coupling clusters (eigenspace)")
    for i, qi in enumerate(c.q_indices):
        w = c.w_dft_by_q[i]
        rel = top_clusters_by_weight(w, threshold=threshold)
        lines.append(f"q[{i}] index={qi}  pos={fmt_q(c.q_positions[i])}")
        lines.append("  cid  size  fmin        fmax        w_dft")
        for cid in rel:
            cl = c.clusters_by_q[i][cid]
            fmin, fmax = c.cluster_ranges_by_q[i][cid]
            lines.append(f"  {cid:3d}  {len(cl):4d}  {fmin: .6f}  {fmax: .6f}  {w[cid]: .6f}")
        lines.append("")

    # Per-ML model comparisons vs DFT
    h("MLIP comparisons")
    for ml_path, res in out.results_per_ml.items():
        lines.append(f"Model: {ml_path}")
        leg_sum = res["AvgProjPowX"]["summary"]
        r_sum = res["research"]["summary"]
        g_sum = res.get("gses", {}).get("summary", None)
        if g_sum is not None:
            lines.append(
                f"  dQ score: mean={g_sum['Score_mean']:.6f}  min={g_sum['Score_min']:.6f}  "
                f"E_freq={g_sum['E_freq_mean']:.6f}  E_freq_rel={g_sum['E_freq_rel_mean']:.6f}  E_vec={g_sum['E_vec_mean']:.6f}"
            )
        lines.append(
            f"  AvgProjPowX coupling modes subspace scores X: mean={leg_sum['X_mean']:.6f}  min={leg_sum['X_min']:.6f}  max={leg_sum['X_max']:.6f}"
        )
        lines.append(
            f"  Coupling cluster (eigenspaces) subspace scores: L1w_mean={r_sum['L1_weights_mean']:.6f}  "
            f"wθ_mean={r_sum['weighted_mean_theta_deg_mean']:.6f}  "
            f"w(1-σ^2)_mean={r_sum['weighted_1_minus_sigma2_mean']:.6f}"
        )

        if "gses" in res:
            lines.append("  Per-q dQ score (summary):")
            for qrow in res["gses"]["per_q"]:
                lines.append(
                    f"    q={fmt_q(np.asarray(qrow['q_position']))}  "
                    f"Score={float(qrow['Score']):.6f}  "
                    f"E_freq={float(qrow['E_freq']):.6f}  "
                    f"E_freq_rel={float(qrow['E_freq_rel']):.6f}  "
                    f"E_vec={float(qrow['E_vec']):.6f}"
                )

        lines.append("  Per-q AvgProjPowX angles (summary):")
        for qrow in res["AvgProjPowX"]["per_q"]:
            sig = np.asarray(qrow["sigma"], dtype=float)
            theta = np.asarray(qrow["theta_deg"], dtype=float)
            sig_min = float(np.min(sig)) if sig.size else float("nan")
            sig_mean = float(np.mean(sig)) if sig.size else float("nan")
            th_mean = float(np.mean(theta)) if theta.size else float("nan")
            th_max = float(np.max(theta)) if theta.size else float("nan")
            warn_d = qrow.get("dft_sum_p_warning", None)
            warn_m = qrow.get("ml_sum_p_warning", None)
            wtxt = ""
            if warn_d or warn_m:
                wtxt = "  WARN:" + (" DFT" if warn_d else "") + (" ML" if warn_m else "")
            lines.append(
                f"    q={fmt_q(np.asarray(qrow['q_position']))}  X={qrow['X']:.6f}  "
                f"k_dft={qrow['k_dft']}  k_ml={qrow['k_ml']}  "
                f"σ_min={sig_min:.6f}  σ_mean={sig_mean:.6f}  "
                f"θ_mean={th_mean:.3f}°  θ_max={th_max:.3f}°{wtxt}"
            )

        lines.append("  Per-q coupling cluster stats:")
        for qrow in res["research"]["per_q"]:
            summ = qrow["summary"]
            lines.append(
                f"    q={fmt_q(np.asarray(qrow['q_position']))}  "
                f"L1w={summ['L1_weights_relevant']:.6f}  "
                f"wθ={summ['weighted_mean_theta_deg_relevant']:.6f}  "
                f"w(1-σ^2)={summ['weighted_1_minus_sigma2_relevant']:.6f}"
            )
            lines.append("      cid  size  fmin        fmax        w_dft     w_ml      θ_mean    θ_max     θ_min")
            for cl in qrow["clusters_relevant"]:
                sig = np.asarray(cl["sigma"], dtype=float)
                sig_min = float(np.min(sig)) if sig.size else float("nan")
                fmin, fmax = cl["freq_range_dft"]
                lines.append(
                    f"      {cl['cluster_id']:3d}  {cl['size_dft']:4d}  {fmin: .6f}  {fmax: .6f}  "
                    f"{cl['w_dft']: .6f}  {cl['w_ml_window']: .6f}  "
                    f"{cl['theta_mean_deg']:7.3f}°  {cl['theta_max_deg']:7.3f}°  {np.cos(sig_min): .6f}"
                )
        lines.append("")
    # Final ranking table (primary: dQ Score_mean; fallback: AvgProjPowX X_mean).
    ranking_rows: List[Tuple[float, float, str]] = []
    for ml_path, res in out.results_per_ml.items():
        g_sum = res.get("gses", {}).get("summary", {})
        leg_sum = res.get("AvgProjPowX", {}).get("summary", {})
        score = float(g_sum.get("Score_mean", float("nan")))
        xmean = float(leg_sum.get("X_mean", float("nan")))
        ranking_rows.append((score, -xmean, ml_path))

    # Sort by Score_mean ascending (lower is better); tie-break by higher X_mean
    ranking_rows.sort(key=lambda t: (np.inf if not np.isfinite(t[0]) else t[0], t[1], t[2]))

    h("FINAL RANKING (lower dQ Score is better)")
    lines.append("rank  Score_mean    E_freq      E_vec       E_freq_rel   X_mean      model")
    for r, (score, neg_xmean, ml_path) in enumerate(ranking_rows, start=1):
        # Print a short model identifier rather than the full band.yaml path.
        # Expected layout: .../results/<model_name>/.../band.yaml
        mp = Path(str(ml_path))
        model_name = mp.parent.parent.parent.parent.name if mp.name == "band.yaml" else mp.name
        res = out.results_per_ml[ml_path]
        g_sum = res.get("gses", {}).get("summary", {})
        leg_sum = res.get("AvgProjPowX", {}).get("summary", {})
        lines.append(
            f"{r:4d}  "
            f"{float(g_sum.get('Score_mean', float('nan'))):10.6f}  "
            f"{float(g_sum.get('E_freq_mean', float('nan'))):9.6f}  "
            f"{float(g_sum.get('E_vec_mean', float('nan'))):9.6f}  "
            f"{float(g_sum.get('E_freq_rel_mean', float('nan'))):11.6f}  "
            f"{float(leg_sum.get('X_mean', float('nan'))):9.6f}  "
            f"{model_name}"
        )
    lines.append("")


    return "\n".join(lines)

 
def _extract_masses(data: Mapping[str, Any], natom: int) -> Optional[List[float]]:
    """Extract atomic masses from a phonopy band.yaml payload. they are stored per 
    'point' in the file, we need an array of them indexed with the indexed atoms.

    Args:
        data (Mapping[str, Any]): Parsed YAML data.
        natom (int): Number of atoms expected.

    Returns:
        Optional[List[float]]: Mass list if available, otherwise None.
    """

    # if there is a flat "mass" list
    m = data.get("mass", None)
    if isinstance(m, list) and len(m) == natom and all(isinstance(x, (int, float)) for x in m):
        return [float(x) for x in m]

    # is its a 'points' list.
    pts = data.get("points", None)
    if isinstance(pts, list) and len(pts) == natom:
        masses: List[float] = []
        for pt in pts:
            if not isinstance(pt, dict) or "mass" not in pt:
                return None
            masses.append(float(pt["mass"]))
        return masses

    # if its an 'atoms' list 
    atoms = data.get("atoms", None) or data.get("atom", None)
    if isinstance(atoms, list) and len(atoms) == natom:
        masses = []
        for a in atoms:
            if isinstance(a, dict) and "mass" in a:
                masses.append(float(a["mass"]))
            else:
                return None
        return masses

    return None

 
def _parse_eigenvector(ev: Any, natom: int) -> List[List[complex]]:
    """Parse a phonopy eigenvector entry into complex vectors.

    Args:
        ev (Any): Eigenvector entry from YAML.
        natom (int): Number of atoms.

    Returns:
        List[List[complex]]: Parsed eigenvector list (natom x 3).
    """

    # Expect one eigenvector per atom
    if not isinstance(ev, list) or len(ev) != natom:
        raise ValueError("Invalid eigenvector shape (expected list of length natom)")
    out: List[List[complex]] = []
    for a in ev:
        # Each atom must have 3 components
        if not isinstance(a, list) or len(a) != 3:
            raise ValueError("Invalid eigenvector atom entry (expected length-3 list)")
        comps: List[complex] = []
        for c in a:
            # each component is [real, imag]
            if not (isinstance(c, list) and len(c) == 2):
                raise ValueError("Invalid eigenvector component (expected [re, im])")
            comps.append(complex(float(c[0]), float(c[1])))
        out.append(comps)
    return out

 
def _parse_floats(line: str, n: int) -> List[float]:
    """Parse at least n floats from a whitespace-delimited line.
    read_poscar helper.
    returns parsed floats.
    """

    toks = line.split()
    if len(toks) < n:
        raise ValueError(f"Expected at least {n} floats in line: '{line}'")
    return [float(toks[i]) for i in range(n)]

 
def _all_int(tokens: Sequence[str]) -> bool:
    """Return True if all tokens can be parsed as integers.
    Helper for 'read_poscar' to decipher element symbols vs counts.
    (vasp can swap)

    Args:
        tokens (Sequence[str]): Tokens to check.

    Returns:
        bool: True if all tokens are integers, else False.
    """

    # True only if every token parses as int
    if not tokens:
        return False
    for t in tokens:
        try:
            int(t)
        except Exception:
            return False
    return True

 
def _build_argparser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    """Build argument parser for CLI usage.

    Args:
        defaults (Dict[str, Any]): Default values for arguments.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """

    # CLI for running comparisons from the command line
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--contcar_gs", default=defaults["contcar_gs"])
    p.add_argument("--contcar_es", default=defaults["contcar_es"])
    p.add_argument("--band_dft", default=defaults["band_dft_path"])
    p.add_argument("--band_ml", action="append", default=None, help="Repeatable: --band_ml path/to/band.yaml")
    p.add_argument("--threshold", type=float, default=defaults["threshold"])
    p.add_argument("--freq_cluster_tol", type=float, default=defaults["freq_cluster_tol"])
    p.add_argument("--freq_window", type=float, default=defaults["freq_window"])
    p.add_argument("--gamma_only", action="store_true", default=defaults["gamma_only"])
    p.add_argument("--alpha", type = float, default = defaults["alpha"])
    # Per-mode weighting used in the GS->ES dQ score (and therefore final ranking).
    p.add_argument(
        "--weight_kind",
        default=defaults["weight_kind"],
        choices=["p", "S", "lambda"],
        help="DFT per-mode weight: p=|a|^2, S=|ω||a|^2, lambda=|ω|^2|a|^2",
    )
    return p

 
def discover_ml_band_paths(results_root: Union[str, Path]) -> List[str]:
    """Discover ML model band.yaml paths under the results directory. Since for this project 
    the band.yamls all exist at Project_dir/results/*model*/raw/plumipy_files/band.yaml
    however, it does this generally by a simple search for key file name 'band.yaml'. 

    Args:
        results_root (Union[str, Path]): Root directory containing model results.

    Returns:
        List[str]: Sorted list of band.yaml paths.
    """

    root = Path(results_root)
    if not root.exists():
        return []
    # Find all band.yaml files under results (each should correspond to a model)
    paths = [p for p in root.rglob("band.yaml") if p.is_file()]
    return [str(p) for p in sorted(paths)]


if __name__ == "__main__":
    # Project root (repo) and auto-discovered model band.yaml paths
    proj_root = Path(__file__).resolve().parent.parent.parent
    results_root = proj_root / "results"
    auto_ml_paths = discover_ml_band_paths(results_root)

    defaults = {
        "contcar_gs": "/home/rnpla/projects/mlip_phonons/test/CBVN/CONTCAR_GS",
        "contcar_es": "/home/rnpla/projects/mlip_phonons/test/CBVN/CONTCAR_ES",
        "band_dft_path": "/home/rnpla/projects/mlip_phonons/test/CBVN/band.yaml",
        "band_ml_paths": auto_ml_paths,
        "q_tol": 1e-4,
        "lattice_tol": 1e-5,
        "threshold": 0.9,
        "freq_cluster_tol": 0.5,
        "freq_window": 0.5,
        "remove_mass_weighted_com": True,
        "gamma_only": True,
        "alpha": 1.3,
        "weight_kind": "S",
    }

    args = _build_argparser(defaults).parse_args()
    band_ml_paths = args.band_ml if args.band_ml is not None else defaults["band_ml_paths"]
    if not band_ml_paths:
        raise ValueError(f"No ML band.yaml files found under {results_root}")

    out = run(
        contcar_gs=args.contcar_gs,
        contcar_es=args.contcar_es,
        band_dft_path=args.band_dft,
        band_ml_paths=band_ml_paths,
        q_tol=defaults["q_tol"],
        lattice_tol=defaults["lattice_tol"],
        threshold=float(args.threshold),
        freq_cluster_tol=float(args.freq_cluster_tol),
        freq_window=float(args.freq_window),
        remove_mass_weighted_com=defaults["remove_mass_weighted_com"],
        gamma_only=bool(args.gamma_only),
        alpha = float(args.alpha),
        weight_kind=str(args.weight_kind),
    )

    report = render_report(
        out,
        threshold=float(args.threshold),
        freq_cluster_tol=float(args.freq_cluster_tol),
        freq_window=float(args.freq_window),
        alpha=float(args.alpha),
        weight_kind=str(args.weight_kind),
    )
    print(report)

    # Also persist the report to disk for later inspection/archiving.
    out_dir = proj_root / "resultsPhonCoupling"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = "phonon_coupling_report"
    i = 0
    while True:
        out_path = out_dir / f"{base}_{i}.txt"
        if not out_path.exists():
            break
        i += 1
    out_path.write_text(report, encoding="utf-8")
