"""Build elemental and pairwise features for the Wagih et al. (2020) segregation dataset.

All elemental properties come from one consistent source (pymatgen's tabulated
periodic-table data), and every mismatch feature is computed from raw physical
values, before any scaling.
"""
import numpy as np
import pandas as pd
from pymatgen.core import Element

# 1 GPa * A^3 = 1e9 Pa * 1e-30 m^3 = 1e-21 J  ->  per mole: 1e-21 * 6.02214e23 J = 0.602214 kJ/mol
GPA_A3_TO_KJ_PER_MOL = 0.602214


def _num(x):
    """pymatgen returns floats, FloatWithUnit objects, or None; convert to float or NaN."""
    if x is None:
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


ASSUMED_POISSON_RATIO = 0.3   # typical metal, used only when an element has no tabulated value
NU_TOLERANCE = 0.1            # max allowed gap between implied and tabulated Poisson's ratio


def _complete_elastic_constants(t, nu_tab):
    """Fill gaps and fix inconsistencies in bulk (K) and shear (G) moduli using isotropic elasticity.

    For an isotropic solid, K, G and Poisson's ratio v are linked:
        G = 3K(1 - 2v) / (2(1 + v))      K = 2G(1 + v) / (3(1 - 2v))
        implied v from K and G = (3K - 2G) / (2(3K + G))
    Rules, applied in order:
      1. K missing, G and v known      -> K from G and v
      2. G missing                     -> G from K and v (v = 0.3 if not tabulated)
      3. G inconsistent with K and v   -> G recomputed from K and v
    Every change is logged in t.attrs["elastic_fixes"].
    """
    fixes = []
    for el in t.index:
        K, G, nu = t.at[el, "bulk_modulus"], t.at[el, "shear_modulus"], nu_tab[el]
        if np.isnan(K) and not np.isnan(G) and not np.isnan(nu):
            t.at[el, "bulk_modulus"] = 2 * G * (1 + nu) / (3 * (1 - 2 * nu))
            fixes.append(f"{el}: bulk modulus missing, derived from G and v = {t.at[el, 'bulk_modulus']:.1f} GPa")
            continue
        v = nu if not np.isnan(nu) else ASSUMED_POISSON_RATIO
        G_from_K = 3 * K * (1 - 2 * v) / (2 * (1 + v))
        if np.isnan(G):
            t.at[el, "shear_modulus"] = G_from_K
            src = "tabulated v" if not np.isnan(nu) else f"assumed v = {ASSUMED_POISSON_RATIO}"
            fixes.append(f"{el}: shear modulus missing, derived from K ({src}) = {G_from_K:.1f} GPa")
        elif not np.isnan(nu):
            implied = (3 * K - 2 * G) / (2 * (3 * K + G))
            if abs(implied - nu) > NU_TOLERANCE:
                t.at[el, "shear_modulus"] = G_from_K
                fixes.append(f"{el}: tabulated G = {G:.0f} GPa implies v = {implied:.2f} vs tabulated {nu:.2f}; "
                             f"replaced with {G_from_K:.1f} GPa from K and v")
    t.attrs["elastic_fixes"] = fixes
    return t


def elemental_table(symbols):
    """Tabulated elemental properties for every symbol, from pymatgen's periodic-table data.

    Atomic (Slater) radius is used for size because it exists for all 25 elements, whereas
    metallic radius is missing for P, Si and Pb. Elastic constants are completed and
    sanity-checked by _complete_elastic_constants.
    """
    import warnings
    rows, nu_tab = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in symbols:
            e = Element(s)
            nu_tab[s] = _num(e.poissons_ratio)
            rows.append({
                "element": s,
                "atomic_mass": _num(e.atomic_mass),
                "atomic_radius": _num(e.atomic_radius),       # Angstrom
                "electronegativity": _num(e.X),               # Pauling
                "melting_point": _num(e.melting_point),       # K
                "bulk_modulus": _num(e.bulk_modulus),         # GPa
                "shear_modulus": _num(e.rigidity_modulus),    # GPa
                "density": _num(e.density_of_solid) / 1000,   # g/cm^3
                "molar_volume": _num(e.molar_volume),         # cm^3/mol
                "group": e.group,
                "period": e.row,
            })
    t = pd.DataFrame(rows).set_index("element")
    return _complete_elastic_constants(t, nu_tab)


def pair_features(df, elem):
    """df has Solvent and Solute columns; elem is the elemental table."""
    v = elem.loc[df["Solvent"]].reset_index(drop=True).add_prefix("solvent_")
    s = elem.loc[df["Solute"]].reset_index(drop=True).add_prefix("solute_")
    out = pd.concat([df.reset_index(drop=True), v, s], axis=1)

    r_v, r_s = out["solvent_atomic_radius"], out["solute_atomic_radius"]
    K_s, G_v = out["solute_bulk_modulus"], out["solvent_shear_modulus"]

    # Size misfit (signed and magnitude)
    out["size_misfit"] = (r_s - r_v) / r_v
    out["abs_size_misfit"] = out["size_misfit"].abs()

    # McLean / Friedel elastic strain energy of an oversized or undersized solute
    # E_el = 24*pi*K_s*G_v*r_s*r_v*(r_s - r_v)^2 / (3*K_s*r_s + 4*G_v*r_v)
    out["elastic_misfit_energy"] = (
        24 * np.pi * K_s * G_v * r_s * r_v * (r_s - r_v) ** 2
        / (3 * K_s * r_s + 4 * G_v * r_v)
    ) * GPA_A3_TO_KJ_PER_MOL

    # Chemical and cohesion differences
    out["electronegativity_diff"] = out["solute_electronegativity"] - out["solvent_electronegativity"]
    out["abs_electronegativity_diff"] = out["electronegativity_diff"].abs()
    out["melting_point_ratio"] = out["solute_melting_point"] / out["solvent_melting_point"]
    out["shear_modulus_ratio"] = out["solute_shear_modulus"] / out["solvent_shear_modulus"]
    out["bulk_modulus_ratio"] = out["solute_bulk_modulus"] / out["solvent_bulk_modulus"]
    out["molar_volume_ratio"] = out["solute_molar_volume"] / out["solvent_molar_volume"]
    out["density_ratio"] = out["solute_density"] / out["solvent_density"]
    out["group_diff"] = out["solute_group"] - out["solvent_group"]
    return out
