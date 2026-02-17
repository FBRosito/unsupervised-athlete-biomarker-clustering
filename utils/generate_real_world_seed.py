"""
UTILITY SCRIPT: Generate Synthetic Real-World Data
------------------------------------------------------------
Generates a mock dataset mimicking the 'dados_base.csv' structure used in Phase 1.
Contains 18 biomarkers + Metadata for 22 athletes.

Usage: python generate_real_world_seed.py
"""

import pandas as pd
import numpy as np
import os

# Configuration
FOLDER = 'database'
FILENAME = 'dataset_real_world_example.csv'
FILE_PATH = os.path.join(FOLDER, FILENAME)

N_ATHLETES = 22
RANDOM_SEED = 42

# Columns based on Table 2 of the paper (18 Biomarkers + Metadata)
# Keeping names in Portuguese to match the notebook's translation pipeline
columns = [
    'Nome', 'Posicao',
    # CK (Creatine Kinase)
    'CPK Pré (U/L)', 'CPK Pós (U/L)', 'CPK 24h (U/L)',
    # LDH (Lactate Dehydrogenase)
    'LDH Pré (U/L)', 'LDH Pós (U/L)', 'LDH 24h (U/L)',
    # CRP (C-Reactive Protein)
    'PCR Pré (mg/L)', 'PCR Pós (mg/L)', 'PCR 24h (mg/L)',
    # Cortisol
    'Cortisol Pré (µg/dL)', 'Cortisol Pós (µg/dL)', 'Cortisol 24h (µg/dL)',
    # Testosterone (Note: Units often vary, simulating nmol/L for conversion logic)
    'Testosterona T Pré (nmol/L)', 'Testosterona T Pós (nmol/L)', 'Testosterona 24h (nmol/L)',
    # Others
    'FC Repouso (bpm)', 'Pressão (mmHG)', 'Oximetria (SpO2)'
]

def generate_data():
    # Ensure the folder exists
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        print(f"Created folder: {FOLDER}")

    np.random.seed(RANDOM_SEED)
    data = {}
    
    for col in columns:
        if col == 'Nome':
            data[col] = [f'Athlete_{i+1:02d}' for i in range(N_ATHLETES)]
        elif col == 'Posicao':
            data[col] = np.random.choice(['Def', 'Mid', 'Fwd'], N_ATHLETES)
        else:
            # Generating plausible physiological values
            if 'CPK' in col or 'LDH' in col:
                val = np.random.uniform(150, 600, N_ATHLETES)
            elif 'PCR' in col:
                val = np.random.uniform(0.1, 8.0, N_ATHLETES)
            elif 'Cortisol' in col:
                val = np.random.uniform(8, 25, N_ATHLETES)
            elif 'Testosterona' in col:
                val = np.random.uniform(10, 35, N_ATHLETES)
            elif 'FC' in col:
                val = np.random.uniform(45, 70, N_ATHLETES)
            elif 'Pressão' in col:
                val = np.random.uniform(110, 130, N_ATHLETES)
            elif 'Oximetria' in col:
                val = np.random.uniform(96, 99, N_ATHLETES)
            else:
                val = np.random.uniform(10, 100, N_ATHLETES)
            
            data[col] = np.round(val, 2)

    df = pd.DataFrame(data)
    df.to_csv(FILE_PATH, index=False)
    print(f"Success! '{FILE_PATH}' created with shape {df.shape}.")

if __name__ == "__main__":
    generate_data()