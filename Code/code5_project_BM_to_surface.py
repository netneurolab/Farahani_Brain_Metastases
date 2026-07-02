"""
###############################################################################

This script SOURCES FreeSurfer inside Python, to do the followings:
  - Project volume → surface (white→pial avg) on MNI09a
  - Visualize on native FS mesh
  - Convert FS surfaces to GIFTI and resample to fsLR-32k
  - Visualize fsLR-32k meshes

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import os, shlex, subprocess, json

#------------------------------------------------------------------------------
# Setting and paths
#------------------------------------------------------------------------------

FREESURFER_HOME = Path("/home/afarahani/freesurfer")
SUBJECTS_DIR    = Path("/home/afarahani/freesurfer/subjects")
SUBJ            = "MNI09a"

# Volume to project
VOL_PATH        = Path("/home/afarahani/Desktop/met_data/Data/BM_files/Melanoma_corrected.nii")

# Path to wb_command)
WB_BIN_DIR      = Path("/home/afarahani/Downloads/workbench/bin_linux64")
WB_CMD          = WB_BIN_DIR / "wb_command"

# Target fsLR spheres (32k)
L_SPHERE_FSLR   = Path("/home/afarahani/Desktop/GA/S1200.L.sphere.32k_fs_LR.surf.gii")
R_SPHERE_FSLR   = Path("/home/afarahani/Desktop/GA/S1200.R.sphere.32k_fs_LR.surf.gii")

# Output folder
OUT_DIR         = Path("./fs_surface_outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

#------------------------------------------------------------------------------
# SetUpFreeSurfer
#------------------------------------------------------------------------------

def import_freesurfer_env(fs_home: Path, subjects_dir: Path):
    """
    Source SetUpFreeSurfer.sh in a clean bash login shell, capture env, and
    update this Python process's os.environ so FreeSurfer tools are available.
    """
    if not (fs_home / "SetUpFreeSurfer.sh").exists():
        raise FileNotFoundError(f"Cannot find {fs_home}/SetUpFreeSurfer.sh")
    bash = (
        f'export FREESURFER_HOME="{fs_home}"; '
        f'export SUBJECTS_DIR="{subjects_dir}"; '
        f'source "{fs_home}/SetUpFreeSurfer.sh" >/dev/null 2>&1; '
        # print env as JSON safely:
        'python3 - << "PY"\n'
        'import os, json; print(json.dumps(dict(os.environ)))\n'
        'PY')
    res = subprocess.run(["bash", "-lc", bash], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to source FreeSurfer env:\n{res.stderr}")

    env = json.loads(res.stdout)
    os.environ.update(env)

# Import FS env into this Python process to we can access freesurfer from python
import_freesurfer_env(FREESURFER_HOME, SUBJECTS_DIR)
SUBJECTS_DIR = Path(os.environ["SUBJECTS_DIR"])

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------

def sh(cmd: str):
    print(">>", cmd)
    subprocess.run(shlex.split(cmd), check=True, env=os.environ)

def out_mgh(hemi: str, vol_path: Path) -> Path:
    base = vol_path.name
    for ext in (".nii.gz", ".nii", ".mgz", ".mgh"):
        base = base.replace(ext, "")
    return OUT_DIR / f"{hemi}.{base}.mgh"

def load_mgh_values(p: Path):
    return np.asarray(nib.load(str(p)).get_fdata()).squeeze()

def load_fs_surf(hemi: str, surf="inflated"):
    from nibabel.freesurfer import read_geometry
    surf_path = SUBJECTS_DIR / SUBJ / "surf" / f"{hemi}.{surf}"
    coords, faces = read_geometry(str(surf_path))
    return coords, faces

def show_stat_on_surf(coords, faces, values, title):
    try:
        from nilearn import plotting
        plotting.plot_surf_stat_map((coords, faces.astype(np.int64)),
                                    stat_map=values, colorbar=True, title=title)
    except Exception:
        ax = plt.figure(figsize=(6,5)).add_subplot(111, projection="3d")
        tri = ax.plot_trisurf(coords[:,0], coords[:,1], coords[:,2],
                              triangles=faces, shade=False, linewidth=0.0)
        tri.set_array(values); tri.autoscale()
        ax.set_title(title); ax.set_axis_off()

def load_gifti_surface(path):
    g = nib.load(str(path))
    coords = g.agg_data('pointset')
    faces  = g.agg_data('triangle')
    return coords, faces

def quick_show_mesh(title, surf_path):
    coords, faces = load_gifti_surface(surf_path)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(coords[:,0], coords[:,1], coords[:,2],
                    triangles=faces, antialiased=True, linewidth=0.0, shade=True)
    ax.set_title(title); ax.set_axis_off(); plt.tight_layout()

#------------------------------------------------------------------------------
# Project volume → surface
#------------------------------------------------------------------------------

for hemi in ("lh", "rh"):
    mgh = out_mgh(hemi, VOL_PATH)
    sh(
        f"mri_vol2surf --mov {VOL_PATH} --regheader {SUBJ} "
        f"--hemi {hemi} --interp trilinear "
        "--projfrac-avg -1 1 0.1 "
       # f"--projfrac -1 "
        f"--out {mgh}"
    )
'''
def single_depth_mask(hemi: str, projflag: str, value: float, out_name: str):
    tmp = OUT_DIR / f"{hemi}.tmp.singledepth.nii.gz"
    sh(
        "mri_surf2vol "
        f"--hemi {hemi} --surf white "
        f"{projflag} {value} "
        f"--subject {SUBJ} --identity {SUBJ} "
        f"--template {SUBJECTS_DIR}/{SUBJ}/mri/T1.mgz "
        f"--surfval {SUBJECTS_DIR}/{SUBJ}/surf/{hemi}.curv "
        f"--o {tmp}"
    )
    # binarize in Python to avoid extra CLI deps
    img = nib.load(str(tmp)); data = img.get_fdata()
    mask = (data > 1e-12).astype(np.uint8)
    outp = OUT_DIR / out_name
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(outp))

    print(f"[ok] wrote mask: {outp}")

# Example: 10% into WM relative to thickness
single_depth_mask("lh", "--projfrac", 0, f"lh.sampled_frac.mask.nii.gz")
single_depth_mask("rh", "--projfrac", 0, f"rh.sampled_frac.mask.nii.gz")
'''
'''
# assumes: SUBJ, SUBJECTS_DIR, OUT_DIR, sh() already defined (from your script)
def sampled_volume_mask_for_projfrac_avg(
    subj: str,
    depths=np.round(np.arange(-1.0, 1.0, 0.1), 2),
    out_prefix="projfrac_avg_-1_to_1_step0p1"
):
    """
    Reproduce the exact sampling domain used by:
      mri_vol2surf ... --projfrac-avg -1 1 0.1
    by unioning the 21 single-depth sheets from mri_surf2vol.

    Writes:
      OUT_DIR/lh.<out_prefix>.mask.nii.gz
      OUT_DIR/rh.<out_prefix>.mask.nii.gz
      OUT_DIR/bh.<out_prefix>.mask.nii.gz  (left ∪ right)
    """
    t1 = Path(os.environ["SUBJECTS_DIR"]) / subj / "mri" / "T1.mgz"
    ref_img = nib.load(str(t1))
    ref_aff = ref_img.affine
    ref_hdr = ref_img.header.copy()
    vol_shape = ref_img.shape

    hemi_masks = {}
    for hemi in ("lh", "rh"):
        acc = np.zeros(vol_shape, dtype=bool)

        for d in depths:
            # tmp file per depth
            tag = str(d).replace("-", "m").replace(".", "p")
            tmp = OUT_DIR / f"{hemi}.tmp.projfrac_{tag}.nii.gz"

            # We just need any per-vertex file to "paint"; lh.curv/rh.curv always exist
            sh(
                "mri_surf2vol "
                f"--hemi {hemi} --surf white "
                f"--projfrac {d} "
                f"--subject {subj} --identity {subj} "
                f"--template {t1} "
                f"--surfval {Path(os.environ['SUBJECTS_DIR'])/subj/'surf'/f'{hemi}.curv'} "
                f"--o {tmp}"
            )

            data = nib.load(str(tmp)).get_fdata()
            acc |= (data > 1e-12)

        # save hemi mask
        out_hemi = OUT_DIR / f"{hemi}.{out_prefix}.mask.nii.gz"
        nib.save(nib.Nifti1Image(acc.astype(np.uint8), ref_aff, ref_hdr), str(out_hemi))
        print(f"[ok] wrote: {out_hemi}")
        hemi_masks[hemi] = acc

    # save bilateral union
    both = (hemi_masks["lh"] | hemi_masks["rh"]).astype(np.uint8)
    out_both = OUT_DIR / f"bh.{out_prefix}.mask.nii.gz"
    nib.save(nib.Nifti1Image(both, ref_aff, ref_hdr), str(out_both))
    print(f"[ok] wrote: {out_both}")


# --- now write the exact volume mask of that sampling domain ---
sampled_volume_mask_for_projfrac_avg(SUBJ)
'''
#------------------------------------------------------------------------------
# Visualization
#------------------------------------------------------------------------------

for hemi in ("lh", "rh"):
    vals = load_mgh_values(out_mgh(hemi, VOL_PATH))
    coords, faces = load_fs_surf(hemi, "inflated") # or "pial"/"white"
    show_stat_on_surf(coords, faces, vals, f"{hemi.upper()} native: {VOL_PATH.name}")
    print(f"{hemi}: n={vals.size}  min={np.nanmin(vals):.4f}  max={np.nanmax(vals):.4f}  mean={np.nanmean(vals):.4f}")
plt.show()

#------------------------------------------------------------------------------
# GIFTI format- yet the subject space - MNI09a
#------------------------------------------------------------------------------

pairs = [
    (f"{SUBJECTS_DIR}/{SUBJ}/surf/lh.white",      "lh.white.native.surf.gii"),
    (f"{SUBJECTS_DIR}/{SUBJ}/surf/lh.pial",       "lh.pial.native.surf.gii"),
    (f"{SUBJECTS_DIR}/{SUBJ}/surf/lh.sphere.reg", "lh.sphere.reg.native.surf.gii"),
    (f"{SUBJECTS_DIR}/{SUBJ}/surf/rh.white",      "rh.white.native.surf.gii"),
    (f"{SUBJECTS_DIR}/{SUBJ}/surf/rh.pial",       "rh.pial.native.surf.gii"),
    (f"{SUBJECTS_DIR}/{SUBJ}/surf/rh.sphere.reg", "rh.sphere.reg.native.surf.gii")]
for src, dst in pairs:
    sh(f"mris_convert {src} {dst}")

#------------------------------------------------------------------------------
# Save vol2surf outputs as GIFTI metrics (.func.gii)
#------------------------------------------------------------------------------

def mgh_to_gifti_func(mgh_path: Path, hemi: str, surf_vertices: int) -> Path:
    """Write a GIFTI metric aligned to the native FS mesh vertex count."""
    vals = np.asarray(nib.load(str(mgh_path)).get_fdata()).squeeze()

    # Let nibabel infer datatype; skip intent to avoid KeyError on older versions
    da = nib.gifti.GiftiDataArray(data=vals.astype(np.float32))

    img = nib.gifti.GiftiImage(darrays=[da])
    img.meta = nib.gifti.GiftiMetaData.from_dict({
        "AnatomicalStructurePrimary": "CortexLeft" if hemi == "lh" else "CortexRight",
        "GeometricType": "FreeSurferNative",
        "Name": mgh_path.stemd})

    out_path = mgh_path.parent / (mgh_path.stem + ".func.gii")
    nib.save(img, str(out_path))
    return out_path

# Get native vertex counts
native_nv = {}
for hemi in ("lh", "rh"):
    coords, faces = load_fs_surf(hemi, "white") # any native FS surf gives the vertex count
    native_nv[hemi] = coords.shape[0]

# Convert both hemispheres
for hemi in ("lh", "rh"):
    mgh = out_mgh(hemi, VOL_PATH)
    out_gii = mgh_to_gifti_func(mgh, hemi, native_nv[hemi])
    print(f"[ok] wrote: {out_gii}")

#------------------------------------------------------------------------------
# Move to fsavegre from subject space - this is to find the tranformation
#------------------------------------------------------------------------------

def fs_surf_nv(subject: str, hemi: str, surf="white") -> int:
    from nibabel.freesurfer import read_geometry
    p = SUBJECTS_DIR / subject / "surf" / f"{hemi}.{surf}"
    v, _ = read_geometry(str(p))
    return v.shape[0]

for hemi in ("lh", "rh"):
    src_mgh = out_mgh(hemi, VOL_PATH) # native metric from vol2surf
    trg_mgh = OUT_DIR / f"{hemi}.{VOL_PATH.stem}.fsaverage.mgh"
    sh(
        f"mri_surf2surf "
        f"--srcsubject {SUBJ} --trgsubject fsaverage "
        f"--hemi {hemi} "
        f"--sval {src_mgh} "
        f"--tval {trg_mgh}")

fsavg_nv = {"lh": fs_surf_nv("fsaverage", "lh", "white"),
            "rh": fs_surf_nv("fsaverage", "rh", "white")}

fsavg_func = {}
for hemi in ("lh", "rh"):
    mgh = OUT_DIR / f"{hemi}.{VOL_PATH.stem}.fsaverage.mgh"
    gii = OUT_DIR / f"{hemi}.{VOL_PATH.stem}.fsaverage.func.gii"
    gii = mgh_to_gifti_func(mgh, hemi, fsavg_nv[hemi])
    fsavg_func[hemi] = gii

#------------------------------------------------------------------------------
# Move fsaverage --> fsLR-32k
#------------------------------------------------------------------------------

L_MID_FSLR = Path("/home/afarahani/Desktop/GA/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii")
R_MID_FSLR = Path("/home/afarahani/Desktop/GA/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii")

path = Path('/home/afarahani/Desktop/MNI_templates_atlases/HCP_templates/standard_mesh_atlases/resample_fsaverage/')
hemi_map = {
    "lh": {
        "src_sphere": path / "fsaverage_std_sphere.L.164k_fsavg_L.surf.gii",
        "trg_sphere": path / "fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii",
        "src_mid":    path / "fsaverage.L.midthickness_va_avg.164k_fsavg_L.shape.gii",
        "trg_mid":    path / "fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii",
    },
    "rh": {
        "src_sphere":path /  "fsaverage_std_sphere.R.164k_fsavg_R.surf.gii",
        "trg_sphere":path /  'fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii',
        "src_mid":   path /  "fsaverage.R.midthickness_va_avg.164k_fsavg_R.shape.gii",
        "trg_mid":   path /  "fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii",
    }
}

fslr_func_32k = {}
for hemi in ("lh", "rh"):
    src_metric = fsavg_func[hemi]
    out_metric = OUT_DIR / f"{hemi}.{VOL_PATH.stem}.32k_fsLR.func.gii"
    sh(
        f"{WB_CMD} -metric-resample "
        f"{src_metric} "
        f"{hemi_map[hemi]['src_sphere']} "
        f"{hemi_map[hemi]['trg_sphere']} "
        f"ADAP_BARY_AREA {out_metric} "
        f"-area-metrics {hemi_map[hemi]['src_mid']} {hemi_map[hemi]['trg_mid']}")
    fslr_func_32k[hemi] = out_metric

'''
#------------------------------------------------------------------------------
# Visualize fsLR-32k result - in python
#------------------------------------------------------------------------------

def show_stat_on_gifti_surf(surf_path: Path, metric_path: Path, title: str):
    s = nib.load(str(surf_path))
    coords = s.agg_data('pointset')
    faces  = s.agg_data('triangle')
    vals   = np.asarray(nib.load(str(metric_path)).darrays[0].data)
    if vals.size != coords.shape[0]:
        raise RuntimeError(f"Size mismatch: {vals.size} != {coords.shape[0]} for {surf_path.name}")
    try:
        from nilearn import plotting
        plotting.plot_surf_stat_map((coords, faces.astype(np.int64)),
                                    stat_map=vals, colorbar=True, title=title)
    except Exception:
        ax = plt.figure(figsize=(6,5)).add_subplot(111, projection="3d")
        tri = ax.plot_trisurf(coords[:,0], coords[:,1], coords[:,2],
                              triangles=faces, shade=False, linewidth=0.0)
        tri.set_array(vals); tri.autoscale()
        ax.set_title(title); ax.set_axis_off()

for hemi, mid_path in (("lh", L_MID_FSLR), ("rh", R_MID_FSLR)):
    show_stat_on_gifti_surf(mid_path, fslr_func_32k[hemi],
                            f"{hemi.upper()} fsLR-32k: {VOL_PATH.stem}")

plt.show()
'''
#------------------------------------------------------------------------------
# END