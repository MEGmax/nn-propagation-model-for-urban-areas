import argparse
import subprocess
import sys
import os
import platform
from pathlib import Path

def get_blender_path():
    """Attempt to find Blender executable."""
    system = platform.system()
    if system == "Darwin":  # macOS
        paths = [
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender 4.2.0.app/Contents/MacOS/Blender", # Add more if needed
        ]
        for p in paths:
            if os.path.exists(p):
                return p
    elif system == "Windows":
        paths = [
            "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe",
             # add more
        ]
        for p in paths:
            if os.path.exists(p):
                return p
    elif system == "Linux":
         # Standard linux paths
         pass
    
    # Fallback: check if 'blender' is in PATH
    try:
        subprocess.run(["blender", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "blender"
    except FileNotFoundError:
        pass
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Run full data generation pipeline.")
    parser.add_argument("--blender-path", type=str, help="Path to Blender executable")
    parser.add_argument("--num-scenes", type=int, default=5, help="Number of scenes to generate")
    parser.add_argument("--skip-rendering", action="store_true", help="Skip the Blender rendering step")
    
    args = parser.parse_args()
    
    # 1. Locate Blender
    blender_exec = args.blender_path
    if not args.skip_rendering:
        if not blender_exec:
            blender_exec = get_blender_path()
        
        if not blender_exec:
            print("Error: Blender executable not found. Please provide path using --blender-path")
            sys.exit(1)
            
        print(f"Using Blender at: {blender_exec}")

        # 2. Run Scene Generation (Blender)
        print("\n--- Step 1: Generating Scenes (Blender) ---")
        scene_gen_script = Path("scene_generation/studio_setup.py").resolve()
        
        if not scene_gen_script.exists():
             print(f"Error: Could not find {scene_gen_script}")
             sys.exit(1)

        cmd = [
            blender_exec,
            "--background",
            "--python", str(scene_gen_script),
            "--",
            "--num-scenes", str(args.num_scenes)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("Scene generation complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error running Blender: {e}")
            sys.exit(1)
    else:
        print("\n--- Step 1: Skipped (User Request) ---")


    # 3. Run Elevation Map Generation
    print("\n--- Step 2: Generating Elevation Maps ---")
    elevation_script = Path("scene_generation/2d_elevation_map.py").resolve()
    
    cmd_elevation = [sys.executable, str(elevation_script)]
    
    try:
        subprocess.run(cmd_elevation, check=True)
        print("Elevation map generation complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error running elevation map script: {e}")
        sys.exit(1)

    # 4. Run Sionna Ray Tracing
    print("\n--- Step 3: Generating RSS Maps (Sionna) ---")
    sionna_script = Path("scene_generation/load_sionna_scene.py").resolve()
    
    cmd_sionna = [sys.executable, str(sionna_script)]
    
    try:
        subprocess.run(cmd_sionna, check=True)
        print("RSS map generation complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error running Sionna script: {e}")
        sys.exit(1)

    print("\n=== Data Generation Pipeline Complete! ===")

if __name__ == "__main__":
    main()
