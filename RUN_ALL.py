#!/usr/bin/env python3
"""
RUN_ALL.py

Execute the complete analysis pipeline.
"""

import subprocess
import sys
import time

import subprocess
import sys
import time

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os

figure_outputs = {
    "make_Figure_1.py": ["Figure_1.png"],
    "make_Figure_2_upper_panel.py": ["Figure_2_upper_panel.png"],
    "make_Figure_2_lower_panel.py": ["Figure_2_lower_panel.png"],
    "make_Figure_3.py": ["Figure_3.png"],
}

t0 = time.perf_counter()

pipeline = [

    ("STEP 1. Convert Pantheon+ covariance matrix", [
        "convert_cov_to_npy.py",
    ]),

    ("STEP 2. Perform cosmological fits", [

        # Flat LCDM
        "fit_LCDM.Pan_original.py",
        "fit_LCDM.DES_original.py",
        "fit_LCDM.Joint_original.py",
        "fit_LCDM.Pan_age_corrected.py",
        "fit_LCDM.DES_age_corrected.py",
        "fit_LCDM.Joint_age_corrected.py",

        # Flat wCDM
        "fit_wCDM.Pan_original.py",
        "fit_wCDM.DES_original.py",
        "fit_wCDM.Joint_original.py",
        "fit_wCDM.Pan_age_corrected.py",
        "fit_wCDM.DES_age_corrected.py",
        "fit_wCDM.Joint_age_corrected.py",

        # Logarithmic relation
        "fit_Loga.Pan_original.py",
        "fit_Loga.DES_original.py",
        "fit_Loga.Joint_original.py",
        "fit_Loga.Pan_age_corrected.py",
        "fit_Loga.DES_age_corrected.py",
        "fit_Loga.Joint_age_corrected.py",

        # Quadratic logarithmic relation
        "fit_Quadratic_Loga.Joint_original.py",
        "fit_Quadratic_Loga.Joint_age_corrected.py",
    ]),

    ("STEP 3. Generate manuscript tables", [
        "make_Table_1.py",
        "make_Table_C1.py",
        "make_Tables_D1_and_D2.py",
        "make_Table_E1.py",
        "make_Tables_G1_and_G2.py",
    ]),

    ("STEP 4. Generate manuscript figures", [
        "make_Figure_1.py",
        "make_Figure_2_upper_panel.py",
        "make_Figure_2_lower_panel.py",
        "make_Figure_3.py",
    ]),
]


for heading, scripts in pipeline:

    print("\n" + "=" * 72)
    print(heading)
    print("=" * 72)

    for script in scripts:

        print(f"\n>>> Running {script}")
        
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"

        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            env=env
        )
        
        if script in figure_outputs:
            for filename in figure_outputs[script]:
                path = Path(filename)
        
                if path.exists():
                    image = mpimg.imread(path)
        
                    plt.figure()
                    plt.imshow(image)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"WARNING: Could not find {filename}")
            
        print(result.stdout, end="")
        
        if result.stderr:
            print(result.stderr, end="")
        
        if result.returncode != 0:
            print(f"\nERROR: {script} failed.")
            sys.exit(result.returncode)

elapsed = time.perf_counter() - t0

print("\n" + "=" * 72)
if elapsed < 60:
    print(f"All scripts completed successfully in {elapsed:.1f} s.")
else:
    print(f"All scripts completed successfully in {elapsed/60:.1f} min.")
print("=" * 72)