"""
pyspin.cluster
==============
Local-cluster ("fragment") approximation — the trick MestReNova uses to scale
to large spin systems. Lets pyspin break past the exact engine's ~15-coupled-
spin wall with near-exact accuracy on sparse graphs, in linear time.

Idea
----
Partition the coupling graph into clusters of ≤ ``max_cluster`` spins by keeping
the strongest couplings intra-cluster and CUTTING the weakest bonds. Within a
cluster, simulate exactly (full second-order, via composite reduction). Treat
each cut (inter-cluster) bond as a classical first-order coupling: the spins on
the far side act as a static Iz "bath" that shifts the cluster's resonances by
J·m for each bath total-Iz value m, weighted by its binomial multiplicity. This
reproduces ordinary first-order multiplet splitting (doublet/triplet/quartet…)
from distant couplings while keeping strong local coupling exact.

Properties
----------
- Exact in the limit of no cut bonds (reduces to ``simulate_spectrum_composite``).
- Exact for the intra-cluster strong coupling; first-order for inter-cluster.
- Cost ~ (#clusters) × (cluster exact cost) × (bath multiplet combinations),
  i.e. linear in molecule size for a bounded cluster size — like MNova.

This is an APPROXIMATION; validate against the exact engine where feasible
(see tests/test_cluster.py).
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from simulation.pyspin.composite import system_transitions
from simulation.pyspin.simulator import peaks_to_spectrum

__all__ = ["partition_clusters", "simulate_spectrum_clustered",
           "simulate_spectrum_pyspin", "clustered_transitions",
           "transitions_pyspin"]


def simulate_spectrum_pyspin(shifts, couplings, degeneracy, field_mhz,
                             exact_max_spins=12, max_cluster=9, **kw):
    """Dispatch: exact composite if the largest coupled fragment is small,
    else the clustered approximation. The wall-free pyspin entry point.

    ``exact_max_spins`` — use the exact engine when the largest connected
    component (spins) is ≤ this; otherwise approximate with clustering.
    """
    from simulation.pyspin.composite import (largest_component_spins,
                                             simulate_spectrum_composite)
    if largest_component_spins(couplings, degeneracy) <= exact_max_spins:
        return simulate_spectrum_composite(shifts, couplings, degeneracy,
                                           field_mhz, **kw)
    return simulate_spectrum_clustered(shifts, couplings, degeneracy, field_mhz,
                                       max_cluster=max_cluster, **kw)


def partition_clusters(couplings, degeneracy, max_cluster):
    """Greedily group spin groups into clusters of ≤ ``max_cluster`` spins.

    Kruskal-style: merge groups across the strongest bonds first, as long as the
    merged cluster stays within ``max_cluster`` spins (degeneracy summed). Bonds
    left uncut-but-crossing become the first-order "cut" bonds.

    Returns a list of clusters, each a sorted list of group indices.
    """
    G = len(degeneracy)
    parent = list(range(G))
    size = [int(degeneracy[g]) for g in range(G)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    for a in range(G):
        for b in range(a + 1, G):
            if couplings[a][b] != 0.0:
                edges.append((abs(couplings[a][b]), a, b))
    edges.sort(reverse=True)  # strongest coupling first

    for _, a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb and size[ra] + size[rb] <= max_cluster:
            parent[ra] = rb
            size[rb] += size[ra]

    clusters = {}
    for g in range(G):
        clusters.setdefault(find(g), []).append(g)
    return [sorted(c) for c in clusters.values()]


def _bath_options(d: int):
    """Total-Iz values m and binomial multiplicities for ``d`` equivalent spin-½."""
    return [(k - d / 2.0, math.comb(d, k)) for k in range(d + 1)]


def clustered_transitions(shifts, couplings, degeneracy, field_mhz,
                          max_cluster=9, intensity_threshold=1e-6):
    """Molecule's approximate line list as ``(centers_ppm, amps)`` via the
    local-cluster method (exact within clusters, first-order Iz bath between)."""
    G = len(shifts)
    clusters = partition_clusters(couplings, degeneracy, max_cluster)

    all_f, all_a = [], []
    for active in clusters:
        aset = set(active)
        active_protons = sum(int(degeneracy[g]) for g in active)

        # External groups cut-coupled to this cluster → classical Iz bath.
        bath = [e for e in range(G)
                if e not in aset and any(couplings[a][e] != 0.0 for a in active)]
        bath_opts = [_bath_options(int(degeneracy[e])) for e in bath]

        sub_coupl = [[couplings[a][b] for b in active] for a in active]
        sub_deg = [degeneracy[g] for g in active]

        cf_list, ca_list = [], []
        combos = product(*bath_opts) if bath_opts else [()]
        for combo in combos:
            weight = 1
            offset = [0.0] * len(active)  # extra Hz offset per active group
            for e, (m, mult) in zip(bath, combo):
                weight *= mult
                for ai, g in enumerate(active):
                    Jge = couplings[g][e]
                    if Jge != 0.0:
                        offset[ai] += Jge * m  # first-order Iz·Iz shift (Hz)
            eff_shifts = [shifts[g] + offset[ai] / field_mhz
                          for ai, g in enumerate(active)]
            f, a = system_transitions(eff_shifts, sub_coupl, sub_deg, field_mhz,
                                      intensity_threshold)
            if len(f):
                cf_list.append(f); ca_list.append(a * weight)

        if not cf_list:
            continue
        cf = np.concatenate(cf_list); ca = np.concatenate(ca_list)
        raw = ca.sum()
        if raw > 0:
            ca = ca * (active_protons / raw)  # areas ∝ proton count
        all_f.append(cf); all_a.append(ca)

    freqs = np.concatenate(all_f) if all_f else np.array([])
    amps = np.concatenate(all_a) if all_a else np.array([])
    return freqs / field_mhz, amps


def simulate_spectrum_clustered(
    shifts,
    couplings,
    degeneracy,
    field_mhz,
    max_cluster=9,
    points=16384,
    ppm_from=0.0,
    ppm_to=12.0,
    linewidth_hz=1.0,
    intensity_threshold=1e-6,
):
    """Clustered (first-order-between-clusters) 1H simulation.

    Returns (ppm_axis, intensity), ∫ = 1.
    """
    centers, amps = clustered_transitions(shifts, couplings, degeneracy, field_mhz,
                                          max_cluster, intensity_threshold)
    ppm = np.linspace(ppm_from, ppm_to, points)
    spec = peaks_to_spectrum(centers, amps, points=points, ppm_from=ppm_from,
                             ppm_to=ppm_to, linewidth_hz=linewidth_hz,
                             field_mhz=field_mhz, normalize=True)
    return ppm, spec


def transitions_pyspin(shifts, couplings, degeneracy, field_mhz,
                       exact_max_spins=12, max_cluster=9, intensity_threshold=1e-6):
    """Line list ``(centers_ppm, amps)`` via the wall-free dispatcher:
    exact composite for small coupled fragments, else clustered."""
    from simulation.pyspin.composite import (composite_transitions,
                                             largest_component_spins)
    if largest_component_spins(couplings, degeneracy) <= exact_max_spins:
        return composite_transitions(shifts, couplings, degeneracy, field_mhz,
                                     intensity_threshold)
    return clustered_transitions(shifts, couplings, degeneracy, field_mhz,
                                 max_cluster, intensity_threshold)
