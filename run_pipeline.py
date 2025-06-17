#!/usr/bin/env python3
# run_pipeline.py
#
# Usage:  
#   (pipeline_env) D:\Facultate\AN4\licenta\D-FINE_and_Hunyuan> python run_pipeline.py --image room.jpg
#
# This script:
#   1. Runs D-FINE’s torch_inf.py (no "-o" override) → crops go to whatever the YAML says.
#   2. Scans the crop folder (as set in the YAML).
#   3. For each crop, runs HUNYUAN-3D shape+paint on GPU → exports .glb.

import os
import sys
import argparse
import subprocess
import glob
import traceback

from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

def run_dfine_inference(
    dfine_root: str,
    config_path: str,
    checkpoint_path: str,
    input_image: str,
    device: str = "cuda:0",
) -> None:
    """
    Launches D-FINE’s torch_inf.py WITHOUT an "-o" flag (it will read output_dir from the YAML).
    """
    torch_inf_py = os.path.join(dfine_root, "tools", "inference", "torch_inf.py")
    if not os.path.isfile(torch_inf_py):
        raise FileNotFoundError(f"Could not find torch_inf.py at: {torch_inf_py}")

    cmd = [
        sys.executable,
        torch_inf_py,
        "-c", config_path,
        "-r", checkpoint_path,
        "-i", input_image,
        "-d", device,
    ]

    print("\n=== Running D-FINE inference ===")
    print("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"D-FINE inference failed (exit code {e.returncode})") from e

    print("→ D-FINE inference completed. Check your YAML’s output_dir for the crops.\n")


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: D-FINE → HUNYUAN-3D for one input image.")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image (e.g. room.jpg)."
    )
    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────────────
    # 1) CONFIGURATION (adjust these if your folder names differ)
    # ──────────────────────────────────────────────────────────────────────────
    PROJECT_ROOT = os.path.dirname(__file__)

    # Path to D-FINE’s repository:
    DFINE_ROOT = os.path.join(PROJECT_ROOT, "D-FINE")

    # D-FINE YAML + checkpoint (relative to DFINE_ROOT)
    DFINE_CONFIG     = os.path.join(DFINE_ROOT, "configs", "dfine", "objects365", "dfine_hgnetv2_x_obj365.yml")
    DFINE_CHECKPOINT = os.path.join(DFINE_ROOT, "checkpoints", "dfine_x_obj365.pth")

    # Input image (user‐supplied via --image)
    INPUT_IMAGE = os.path.join(PROJECT_ROOT, args.image)
    if not os.path.isfile(INPUT_IMAGE):
        print(f"Error: Input image not found: {INPUT_IMAGE}")
        sys.exit(1)

    # The folder where D-FINE writes all the cropped objects.
    #  ↳ This must match "output_dir:" (or "crop_folder:") inside your YAML exactly.
    CROP_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "crops", "room")

    # Where we’ll dump the final .glb meshes
    OUTPUT_3D_ROOT = os.path.join(PROJECT_ROOT, "outputs", "3d_meshes")

    # Which GPU to use
    DEVICE = "cuda:0"

    # HUNYUAN pretrained model identifiers
    HUNYUAN_SHAPEDIR = "tencent/Hunyuan3D-2"
    HUNYUAN_PAINTDIR = "tencent/Hunyuan3D-2"

    # ──────────────────────────────────────────────────────────────────────────
    # 2) Check that the crop folder exists (YAML should create it when inference runs).
    # ──────────────────────────────────────────────────────────────────────────
    if not os.path.isdir(CROP_FOLDER):
        print(
            f"Error: Expected crop folder doesn’t exist:\n  {CROP_FOLDER}\n"
            f"→ Make sure 'output_dir' in your D-FINE YAML is EXACTLY '{CROP_FOLDER}', or adjust CROP_FOLDER here."
        )
        sys.exit(1)

    # Ensure 3D output directory exists
    os.makedirs(OUTPUT_3D_ROOT, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 3) Run D-FINE inference (cropping step)
    # ──────────────────────────────────────────────────────────────────────────
    try:
        run_dfine_inference(
            dfine_root=DFINE_ROOT,
            config_path=DFINE_CONFIG,
            checkpoint_path=DFINE_CHECKPOINT,
            input_image=INPUT_IMAGE,
            device=DEVICE,
        )
    except Exception as e:
        print("✖ D-FINE inference failed. Aborting pipeline.\n", str(e))
        traceback.print_exc()
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────────
    # 4) Gather all crops from CROP_FOLDER
    # ──────────────────────────────────────────────────────────────────────────
    crop_patterns = ["*.jpg", "*.jpeg", "*.png"]
    all_crops = []
    for pat in crop_patterns:
        all_crops.extend(glob.glob(os.path.join(CROP_FOLDER, pat)))

    if len(all_crops) == 0:
        print(f"⚠️  No cropped images found in {CROP_FOLDER}. Exiting.")
        sys.exit(0)

    print(f"✅ Found {len(all_crops)} cropped-object images. Proceeding to HUNYUAN-3D for each one.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 5) Initialize HUNYUAN‐3D pipelines on GPU
    # ──────────────────────────────────────────────────────────────────────────
    try:
        print("Loading HUNYUAN-3D DiTFlowMatching (shape) pipeline on", DEVICE, "…")
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        HUNYUAN_SHAPEDIR
        ).to(DEVICE)
        print("Loading HUNYUAN-3D Paint pipeline on", DEVICE, "…")
        paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        HUNYUAN_PAINTDIR
        )
    except Exception as e:
        print("✖ Failed to initialize HUNYUAN-3D pipelines. Aborting.")
        traceback.print_exc()
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────────
    # 6) For each crop: run shape → paint → export .glb
    # ──────────────────────────────────────────────────────────────────────────
    for idx, crop_path in enumerate(sorted(all_crops)):
        base_name = os.path.splitext(os.path.basename(crop_path))[0]
        print(f"\n--- [{idx + 1}/{len(all_crops)}] Processing crop: {base_name} ---")

        out_subdir = os.path.join(OUTPUT_3D_ROOT, base_name)
        os.makedirs(out_subdir, exist_ok=True)

        try:
            # a) Shape pipeline
            print("  • Running HUNYUAN shape pipeline…")
            shape_results = shape_pipeline(image=crop_path)  # returns a list
            mesh = shape_results[0]

            # b) Paint pipeline
            print("  • Running HUNYUAN paint pipeline…")
            paint_results = paint_pipeline(mesh, image=crop_path)
            textured_mesh = paint_results[0]

            # c) Export to .glb
            glb_outpath = os.path.join(out_subdir, f"{base_name}.glb")
            print(f"  • Exporting textured mesh to: {glb_outpath}")
            if hasattr(textured_mesh, "export"):
                textured_mesh.export(glb_outpath)
            else:
                # Fallback: try trimesh export
                import trimesh
                if isinstance(textured_mesh, trimesh.Trimesh):
                    textured_mesh.export(glb_outpath)
                else:
                    raise AttributeError(
                        "Returned mesh object has no .export() and is not a trimesh.Trimesh."
                    )

            print(f"  → Successfully wrote {glb_outpath}")

        except Exception as crop_e:
            print(f"  ✖ FAILED on crop {base_name}: {crop_e}")
            traceback.print_exc()
            # Continue with the next crop
            continue

    print("\n=== Pipeline complete! All crops processed. ===")
    print(f"Final GLB files live under:\n  {OUTPUT_3D_ROOT}\n")


if __name__ == "__main__":
    main()
