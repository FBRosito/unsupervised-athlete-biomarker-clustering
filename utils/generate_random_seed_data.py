"""
UTILITY SCRIPT: Generate Synthetic Seed Data
-------------------------------------------
This script generates a 'mock' dataset with the exact structure (columns and units)
of the real athlete data used in the study.

Purpose:
1. Provide a safe, anonymized input for the Jupyter Notebooks.
2. Ensure no real athlete data is shared publicly (Privacy/GDPR compliance).
3. Allow researchers to reproduce the pipeline with a controlled random seed.

Usage:
    python generate_random_seed_data.py
    
Output:
    dataset_example_structure.csv
"""

import pandas as pd
import numpy as np
import os

# Configuration
FOLDER = 'database'
FILENAME = 'dataset_example_structure.csv' # Arquivo da Fase 2 (32 vars)
FILE_PATH = os.path.join(FOLDER, FILENAME)

N_ATHLETES = 15
RANDOM_SEED = 42

# List of Biomarkers (Must match the study's feature set)
columns = [
    'Nome', 'Posicao', 'Numero', 
    'LDH Pós (U/L)', 'Testosterona (nmol/L)', 'Testosterona 24h (nmol/L)', 
    'TGO (U/L)', 'CPK Pré (U/L)', 'LDH (U/L)', 'CPK 24h (U/L)', 'CPK (U/L)', 
    'TGP (U/L)', 'PCR Pós (mg/L)', 'Glicose 24h (mg/dL)', 'Pressão (mmHG)', 
    'Cortisol 24h (µg/dL)', 'Cortisol (U/L)', 'PCR Pré (mg/L)', 
    'Cortisol Pré (µg/dL)', 'CK-MB (U/L)', 'Glicose (mg/dL)', 'FC Repouso (bpm)', 
    'LDH 24h (U/L)', 'PCR 24h Pós (mg/L)', 'Oximetria (SpO2)', 
    'Cortisol Pós (µg/dL)', 'Insulina (µU/mL)', 'Testosterona T Pré (nmol/L)', 
    'CPK Pós (U/L)', 'Homocisteína pré (µmol/L)', 'Homocisteína pós (µmol/L)', 
    'Homocisteína 24h (µmol/L)'
]

def generate_data():
    # Ensure the folder exists
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    np.random.seed(RANDOM_SEED)
    data = {}

    print(f"Generating mock data for {N_ATHLETES} athletes...")

    for col in columns:
        # Categorical / Metadata
        if col == 'Nome':
            data[col] = [f'Example_Athlete_{i+1:02d}' for i in range(N_ATHLETES)]
        elif col == 'Posicao':
            data[col] = np.random.choice(['Defender', 'Midfielder', 'Forward', 'Goalkeeper'], N_ATHLETES)
        elif col == 'Numero':
            data[col] = np.random.randint(1, 99, N_ATHLETES)
        
        # Numerical Biomarkers (Simulating physiological ranges)
        else:
            if 'CPK' in col or 'LDH' in col:
                val = np.random.uniform(100, 600, N_ATHLETES) # Enzymes
            elif 'Testosterona' in col:
                val = np.random.uniform(10, 35, N_ATHLETES)   # nmol/L range
            elif 'Cortisol' in col:
                val = np.random.uniform(5, 25, N_ATHLETES)
            elif 'Glicose' in col:
                val = np.random.uniform(70, 110, N_ATHLETES)
            elif 'Insulina' in col:
                val = np.random.uniform(2, 20, N_ATHLETES)
            elif 'Homocisteína' in col:
                val = np.random.uniform(5, 15, N_ATHLETES)
            elif 'PCR' in col:
                val = np.random.uniform(0.1, 5.0, N_ATHLETES)
            elif 'Pressão' in col:
                val = np.random.uniform(110, 130, N_ATHLETES) 
            elif 'SpO2' in col:
                val = np.random.uniform(95, 99, N_ATHLETES)
            elif 'FC' in col:
                val = np.random.uniform(45, 65, N_ATHLETES)
            else:
                val = np.random.uniform(10, 100, N_ATHLETES) # Fallback for others
                
            data[col] = np.round(val, 2)

    df = pd.DataFrame(data)
    df.to_csv(FILE_PATH, index=False)
    print(f"Success! '{FILE_PATH}' created with shape {df.shape}.")

if __name__ == "__main__":
    generate_data()
