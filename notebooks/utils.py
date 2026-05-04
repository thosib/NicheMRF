"""Shared helpers for benchmark notebooks.


"""

from __future__ import annotations

import json
import os
import sys
import types
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path / environment resolution
# ---------------------------------------------------------------------------

def _is_sdata_root(p: str) -> bool:
    return os.path.isdir(p) and (
        os.path.exists(os.path.join(p, ".zgroup"))
        or os.path.exists(os.path.join(p, ".zmetadata"))
        or os.path.exists(os.path.join(p, "tables"))
    )


def _candidate_zarr_paths(data_root: str, slide_id: str):
    # Probed in priority order. Cleaned outputs from
    # convert_sthelar_to_modern_zarr.py live under a `_clean` suffix and are
    # the preferred source when present.
    return [
        os.path.join(data_root, f"sdata_{slide_id}_clean"),
        os.path.join(data_root, f"sdata_{slide_id}_clean.zarr"),
        os.path.join(data_root, f"sdata_{slide_id}.zarr"),
        os.path.join(data_root, f"sdata_{slide_id}", f"sdata_{slide_id}.zarr"),
        os.path.join(data_root, f"sdata_{slide_id}"),
    ]


def setup_env(
    tool: str,
    slide_id: str,
    data_root_override: str | None = None,
    art_root_override: str | None = None,
) -> Tuple[str, str, str]:
    """Detect Colab vs server; resolve the zarr path and artifact dirs."""
    in_colab = False
    try:
        import google.colab  # noqa: F401
        in_colab = True
    except ImportError:
        pass

    if data_root_override:
        data_root = data_root_override
        art_root = art_root_override or os.path.join(
            os.path.dirname(data_root_override), "artifacts"
        )
    elif in_colab:
        from google.colab import drive
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
        base = "/content/drive/MyDrive/CBMF4761"
        data_root = os.path.join(base, "data")
        art_root = os.path.join(base, "artifacts")
    else:
        base = os.path.abspath(os.path.join(os.getcwd(), ".."))
        data_root = os.path.join(base, "data")
        art_root = os.path.join(base, "artifacts")

    h5ad_dir = os.path.join(art_root, "h5ad")
    fig_dir = os.path.join(art_root, "figures", tool)
    for d in (h5ad_dir, fig_dir):
        os.makedirs(d, exist_ok=True)

    candidates = _candidate_zarr_paths(data_root, slide_id)
    zarr_path = candidates[0]
    for cand in candidates:
        if _is_sdata_root(cand):
            zarr_path = cand
            break

    h5ad_out = os.path.join(h5ad_dir, f"{tool}_{slide_id}.h5ad")
    env_label = "colab" if in_colab else ("override" if data_root_override else "server")
    print(f"env      : {env_label}")
    print(f"zarr     : {zarr_path}  (exists: {os.path.exists(zarr_path)})")
    if not os.path.exists(zarr_path) and not os.path.exists(os.path.dirname(zarr_path)):
        print(f"[setup_env] WARNING: data root not found. On Colab, ensure STHELAR zarr archives are uploaded to Google Drive at {os.path.dirname(zarr_path)} or set DATA_ROOT_OVERRIDE.")
    print(f"h5ad_out : {h5ad_out}")
    print(f"fig_dir  : {fig_dir}")
    return zarr_path, h5ad_out, fig_dir


def save_fig(name: str, fig_dir: str, slide_id: str) -> None:
    import matplotlib.pyplot as plt
    path = os.path.join(fig_dir, f"{slide_id}_{name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("saved:", path)


# ---------------------------------------------------------------------------
# load_table() — five-strategy fallback for STHELAR zarrs
# ---------------------------------------------------------------------------

def _ls(directory: str):
    if not os.path.isdir(directory):
        return set()
    return {e for e in os.listdir(directory) if not e.startswith(".")}


def _to_str(arr):
    if hasattr(arr, "dtype") and arr.dtype.kind in ("S", "O", "U"):
        return np.array(
            [
                x.decode("utf-8", errors="replace") if isinstance(x, bytes)
                else ("" if x is None else str(x))
                for x in arr.flat
            ],
            dtype=str,
        ).reshape(arr.shape)
    return arr


def _is_zarr_array(obj) -> bool:
    import zarr
    return isinstance(obj, zarr.Array) if hasattr(zarr, "Array") else (
        hasattr(obj, "shape") and hasattr(obj, "dtype") and not hasattr(obj, "keys")
    )


def _read_sparse(grp):
    from scipy import sparse as _sp
    attrs = dict(grp.attrs)
    data = np.asarray(grp["data"])
    idx = np.asarray(grp["indices"])
    indptr = np.asarray(grp["indptr"])
    cls = _sp.csc_matrix if "csc" in attrs.get("encoding-type", "") else _sp.csr_matrix
    return cls((data, idx, indptr), shape=tuple(attrs["shape"]))


def _read_dataframe(grp, fs_dir):
    zattrs_path = os.path.join(fs_dir, ".zattrs")
    if not os.path.exists(zattrs_path):
        return pd.DataFrame()
    with open(zattrs_path) as fh:
        attrs = json.load(fh)

    idx_key = attrs.get("_index")
    cols = attrs.get("column-order", [])
    on_disk = _ls(fs_dir)

    data = {}
    for col in cols:
        if col not in on_disk:
            continue
        try:
            obj = grp[col]
            if _is_zarr_array(obj):
                data[col] = _to_str(np.asarray(obj))
            else:
                data[col] = pd.Categorical.from_codes(
                    np.asarray(obj["codes"]),
                    categories=_to_str(np.asarray(obj["categories"])),
                )
        except Exception:
            continue

    df = pd.DataFrame(data)
    if idx_key and idx_key in on_disk:
        try:
            df.index = _to_str(np.asarray(grp[idx_key]))
        except Exception:
            pass
    return df


def _manual_read_zarr_table(zarr_root: str, table_name: str):
    import zarr
    import anndata as ad
    try:
        root = zarr.open_group(zarr_root, mode="r", zarr_format=2, use_consolidated=True)
    except TypeError:
        root = zarr.open_group(zarr_root, mode="r")
    tbl = root["tables"][table_name]
    tbl_fs = os.path.join(zarr_root, "tables", table_name)

    X_obj = tbl["X"]
    X = np.asarray(X_obj) if _is_zarr_array(X_obj) else _read_sparse(X_obj)
    obs = _read_dataframe(tbl["obs"], os.path.join(tbl_fs, "obs"))
    var = _read_dataframe(tbl["var"], os.path.join(tbl_fs, "var"))

    obsm, obsp = {}, {}
    for slot_name, dest in [("obsm", obsm), ("obsp", obsp)]:
        slot_dir = os.path.join(tbl_fs, slot_name)
        for key in _ls(slot_dir):
            try:
                obj = tbl[slot_name][key]
                dest[key] = np.asarray(obj) if _is_zarr_array(obj) else _read_sparse(obj)
            except Exception:
                continue

    return ad.AnnData(
        X=X, obs=obs, var=var,
        obsm=obsm or None, obsp=obsp or None,
    )


def load_table(zarr_root: str, table_name: str):
    """Load a single AnnData table from a STHELAR zarr via 5-strategy fallback.

    Clean stores produced by convert_sthelar_to_modern_zarr.py succeed on
    Strategy 1. Legacy STHELAR stores may fall through to manual reconstruction.
    """
    import anndata as ad

    # Strategy 1: sd.read_zarr with table selection
    try:
        import spatialdata as sd
        sdata = sd.read_zarr(zarr_root, selection=("tables",))
        if table_name in sdata.tables:
            print(f"[load_table] Strategy 1 OK: sd.read_zarr(selection=('tables',))")
            return sdata.tables[table_name]
    except Exception as exc:
        print(f"[load_table] Strategy 1 failed: {type(exc).__name__}: {exc}")

    # Strategy 2: legacy sd.read_zarr
    try:
        import spatialdata as sd
        sdata = sd.read_zarr(zarr_root)
        if table_name in sdata.tables:
            print(f"[load_table] Strategy 2 OK: sd.read_zarr(legacy)")
            return sdata.tables[table_name]
    except Exception as exc:
        print(f"[load_table] Strategy 2 failed: {type(exc).__name__}: {exc}")

    # Strategy 3: anndata.read_zarr on table sub-path
    try:
        tbl_path = os.path.join(zarr_root, "tables", table_name)
        adata = ad.read_zarr(tbl_path)
        print(f"[load_table] Strategy 3 OK: ad.read_zarr(sub-path)")
        return adata
    except Exception as exc:
        print(f"[load_table] Strategy 3 failed: {type(exc).__name__}: {exc}")

    # Strategy 4: zarr.DirectoryStore + anndata.experimental.read_elem
    try:
        import zarr
        from anndata.experimental import read_elem
        tbl_path = os.path.join(zarr_root, "tables", table_name)
        store = zarr.DirectoryStore(tbl_path) if hasattr(zarr, "DirectoryStore") else tbl_path
        try:
            grp = zarr.open_consolidated(store, mode="r")
        except Exception:
            grp = zarr.open_group(store, mode="r")
        slots = {"X": None, "obs": None, "var": None}
        for slot in list(slots):
            try:
                slots[slot] = read_elem(grp[slot])
            except Exception:
                slots[slot] = None
        obsm = {}
        if "obsm" in grp:
            for k in grp["obsm"]:
                try:
                    obsm[k] = read_elem(grp["obsm"][k])
                except Exception:
                    continue
        obsp = {}
        if "obsp" in grp:
            for k in grp["obsp"]:
                try:
                    obsp[k] = read_elem(grp["obsp"][k])
                except Exception:
                    continue
        adata = ad.AnnData(
            X=slots["X"],
            obs=slots["obs"] if slots["obs"] is not None else pd.DataFrame(),
            var=slots["var"] if slots["var"] is not None else pd.DataFrame(),
            obsm=obsm or None,
            obsp=obsp or None,
        )
        print(f"[load_table] Strategy 4 OK: zarr.DirectoryStore + read_elem")
        return adata
    except Exception as exc:
        print(f"[load_table] Strategy 4 failed: {type(exc).__name__}: {exc}")

    # Strategy 5: manual reconstruction (phantom-column-safe)
    print(f"[load_table] Strategy 5: manual reconstruction")
    return _manual_read_zarr_table(zarr_root, table_name)


# ---------------------------------------------------------------------------
# Compatibility shims — call apply_shims() once before importing scanpy/squidpy
# ---------------------------------------------------------------------------

def apply_shims() -> None:
    """Install runtime shims so legacy scanpy/squidpy imports succeed on a
    modern numpy / anndata / IPython stack. Idempotent."""
    # NumPy 2.0 removed these deprecated aliases.
    for name, target in (
        ("float_", "float64"),
        ("int_", "int64"),
        ("complex_", "complex128"),
        ("bool_", "bool_"),
    ):
        if not hasattr(np, name):
            setattr(np, name, getattr(np, target))

    # IPython.display.set_matplotlib_formats was removed in IPython 8.24+.
    try:
        import IPython.display
        if not hasattr(IPython.display, "set_matplotlib_formats"):
            try:
                from matplotlib_inline.backend_inline import set_matplotlib_formats
                IPython.display.set_matplotlib_formats = set_matplotlib_formats
            except ImportError:
                IPython.display.set_matplotlib_formats = lambda *a, **k: None
    except ImportError:
        pass

    # squidpy<1.7 imports anndata.io.read_text; anndata>=0.11 dropped it.
    import anndata
    if not hasattr(anndata, "io"):
        try:
            mod = types.ModuleType("anndata.io")
            _read_text_func = None
            try:
                import anndata._io.read as _ad_read
                _read_text_func = _ad_read.read_text
            except ImportError:
                try:
                    import anndata._io.text as _ad_text
                    _read_text_func = _ad_text.read_text
                except ImportError:
                    pass
            if _read_text_func:
                mod.read_text = _read_text_func
            else:
                mod.read_text = lambda *a,**k: None
                
            sys.modules["anndata.io"] = mod
            anndata.io = mod
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_spatial_connectivity(adata, niche_key, obsp_key='spatial_connectivities') -> float:
    """Calculate the average fraction of same-niche spatial neighbors.
    
    A higher score indicates more spatially contiguous and well-defined niches.
    """
    import numpy as np
    if obsp_key not in adata.obsp:
        raise KeyError(f"{obsp_key} not found in adata.obsp. Run sq.gr.spatial_neighbors first.")
    
    adj = adata.obsp[obsp_key]
    labels = adata.obs[niche_key].values
    # Handle pandas Categorical or object arrays
    if hasattr(labels, 'codes'):
        labels = labels.codes
    
    n_cells = adj.shape[0]
    fractions = []
    
    # Iterate over sparse rows to find neighbors for each cell
    for i in range(n_cells):
        start, end = adj.indptr[i], adj.indptr[i+1]
        neighbors = adj.indices[start:end]
        if len(neighbors) == 0:
            continue
        
        # Calculate fraction of neighbors sharing the same label
        same_niche_count = np.sum(labels[neighbors] == labels[i])
        fractions.append(same_niche_count / len(neighbors))
    
    return float(np.mean(fractions)) if fractions else 0.0


def compute_localized_entropy(adata, niche_key, obsp_key='spatial_connectivities') -> float:
    """Calculate the average Shannon entropy of niche labels in spatial neighborhoods.
    
    A lower score indicates more local homogeneity (fewer mixing of niches).
    """
    import numpy as np
    from scipy.stats import entropy
    if obsp_key not in adata.obsp:
        raise KeyError(f"{obsp_key} not found in adata.obsp. Run sq.gr.spatial_neighbors first.")
    
    adj = adata.obsp[obsp_key]
    labels = adata.obs[niche_key].values
    
    n_cells = adj.shape[0]
    entropies = []
    
    for i in range(n_cells):
        start, end = adj.indptr[i], adj.indptr[i+1]
        neighbors = adj.indices[start:end]
        if len(neighbors) == 0:
            continue
        
        # Get distribution of labels in the neighborhood
        neighbor_labels = labels[neighbors]
        _, counts = np.unique(neighbor_labels, return_counts=True)
        probs = counts / counts.sum()
        entropies.append(entropy(probs))
        
    return float(np.mean(entropies)) if entropies else 0.0
