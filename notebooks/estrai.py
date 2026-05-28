import os
import zipfile
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
from io import BytesIO

# --- CONFIGURAZIONE ---
ZIP_PATH = "home/andy/DriveTesi/vinbigdata-chest-xray-abnormalities-detection.zip"
CSV_PATH = '/home/andy/Documenti/Tesi-Magistrale/data/train.csv'        
OUTPUT_DIR = '/home/andy/Documenti/Tesi-Magistrale/data/RX_super' # Puoi rinominarla in RX_dataset se preferisci
TARGET_SAMPLES = 5000

def run_extraction():
    os.makedirs(os.path.join(OUTPUT_DIR, 'sani'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'malati'), exist_ok=True)

    print("Analisi rigorosa del CSV VinBigData in corso...")
    df = pd.read_csv(CSV_PATH)


    pos_image_ids = df[df['class_name'].isin(['Nodule/Mass'])]['image_id'].unique()

    all_abnormal_ids = df[df['class_name'] != 'No finding']['image_id'].unique()
    all_no_finding_ids = df[df['class_name'] == 'No finding']['image_id'].unique()
    pure_neg_image_ids = np.setdiff1d(all_no_finding_ids, all_abnormal_ids)

    print(f"Trovati {len(pos_image_ids)} Positivi (Nodule/Mass) unici.")
    print(f"Trovati {len(pure_neg_image_ids)} Negativi Puri unici.")

    n_campioni = min(TARGET_SAMPLES, len(pos_image_ids), len(pure_neg_image_ids))
    print(f"\nCampiono {n_campioni} immagini per classe (Totale: {n_campioni * 2})")

    np.random.seed(42)
    final_pos_ids = np.random.choice(pos_image_ids, n_campioni, replace=False)
    final_neg_ids = np.random.choice(pure_neg_image_ids, n_campioni, replace=False)

    da_estrarre = [(img_id, 'malati') for img_id in final_pos_ids] + \
                  [(img_id, 'sani') for img_id in final_neg_ids]

    print(f"\nInizio Estrazione DICOM e Conversione JPEG2000 (Lossless)...")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        for i, (img_id, cartella) in enumerate(da_estrarre, 1):
            path_nello_zip = f"train/{img_id}.dicom" 
            out_path = os.path.join(OUTPUT_DIR, cartella, f"{img_id}.jp2")
            
            if os.path.exists(out_path):
                continue
                
            try:
                with zip_ref.open(path_nello_zip) as file_dicom:
                    dcm = pydicom.dcmread(BytesIO(file_dicom.read()))
                    img_array = dcm.pixel_array.astype(float)
                    
                    # Normalizzazione dei pixel a 8-bit (0-255)
                    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-7) * 255.0
                    img_uint8 = img_array.astype(np.uint8)
                    
                    # Correzione del contrasto (se il radiogramma è invertito nativamente)
                    if getattr(dcm, 'PhotometricInterpretation', '') == "MONOCHROME1":
                        img_uint8 = 255 - img_uint8
                    
                    # Convertiamo in immagine PIL
                    # Usiamo RGB per compatibilità futura con modelli pre-addestrati (come ResNet)
                    final_img = Image.fromarray(img_uint8).convert('RGB')
                    
                    # Salvataggio JPEG2000 senza perdita di dati (Lossless)
                    final_img.save(out_path, format='JPEG2000', quality_mode='dB', quality_vals=[0])
                    
                    if i % 100 == 0 or i == len(da_estrarre):
                        print(f"Progresso: {i}/{len(da_estrarre)} immagini salvate...")
                    
            except Exception as e:
                print(f"Errore su {img_id}: {e}")
                
    print("\nEstrazione e conversione completate con successo!")

if __name__ == "__main__":
    run_extraction()