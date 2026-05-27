

def extract_features(img):
    features = {}
    pixel_vals = img.ravel()
    features['hist_mean'] = np.mean(pixel_vals)
    features['hist_std'] = np.std(pixel_vals)
    features['hist_skew'] = skew(pixel_vals)
    features['hist_kurtosis'] = kurtosis(pixel_vals)
    percentiles = np.percentile(pixel_vals, [5, 25, 50, 75, 95])
    features['perc_05'], features['perc_25'], features['perc_50'], features['perc_75'], features['perc_95'] = percentiles
    h, w = img.shape
    h_step, w_step = h // 3, w // 3
    border_means, center_mean = [], 0
    for i in range(3):
        for j in range(3):
            quadrant = img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            q_mean = np.mean(quadrant)
            features[f'quad_{i}_{j}_mean'] = q_mean
            features[f'quad_{i}_{j}_std'] = np.std(quadrant)
            if i == 1 and j == 1: center_mean = q_mean
            else: border_means.append(q_mean)
    features['border_center_ratio'] = np.mean(border_means) / (center_mean + 1e-5)
    features['shannon_entropy'] = shannon_entropy(img)
    edges = cv2.Canny(img, 50, 150)
    features['edge_density'] = np.sum(edges > 0) / (h * w)
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    features['glcm_energy'] = graycoprops(glcm, 'energy')[0, 0]
    features['glcm_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()
    for idx, hm in enumerate(hu_moments):
        features[f'hu_moment_{idx}'] = -np.sign(hm) * np.log10(np.abs(hm) + 1e-10)
    return features

def process_perfect_image(img):
    rows = []
    rows.append({**extract_features(img), 'class_label': 'Perfetta'})
    rows.append({**extract_features(255 - img), 'class_label': 'Negativo'})
    rows.append({**extract_features(cv2.normalize(img, None, 80, 150, cv2.NORM_MINMAX)), 'class_label': 'Contrast_Stretching'})
    gamma = 0.3
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    rows.append({**extract_features(cv2.LUT(img, table)), 'class_label': 'Gray_Level_Slicing'})
    if random.random() < 0.35:
        gauss = np.random.normal(0, 50, (512, 512))
        rows.append({**extract_features(np.clip(img + gauss, 0, 255).astype(np.uint8)), 'class_label': 'Inutilizzabile'})
    return rows

# ==========================================
# 3. ESECUZIONE (Diretta, niente proxy!)
# ==========================================
dataset_rows = []

print(f"Connesso a Google Drive tramite Ubuntu GVFS: {DRIVE_ROOT}")

if TAR_MIMIC.exists():
    print("Estrazione MIMIC in corso (Lettura Streaming)...")
    df_mimic = pd.merge(pd.read_csv(str(CSV_MIMIC_META)), pd.read_csv(str(CSV_MIMIC_CHEX)), on=['subject_id', 'study_id']).set_index('dicom_id')
    
    with tarfile.open(str(TAR_MIMIC), 'r') as archive:
        immagini = [f for f in archive.getnames() if f.lower().endswith('.jpg')]
        conteggio = 0
        for path in tqdm(immagini, desc="MIMIC"):
            if conteggio >= 1200: break
            dicom_id = os.path.basename(path).replace(".jpg", "")
            if dicom_id in df_mimic.index:
                row = df_mimic.loc[dicom_id]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                if (row['ViewPosition'] == 'PA') and (row['No Finding'] == 1.0) and (row['Support Devices'] != 1.0):
                    file_obj = archive.extractfile(path)
                    if file_obj:
                        img = cv2.imdecode(np.frombuffer(file_obj.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                            dataset_rows.extend(process_perfect_image(img))
                            conteggio += 1

if ZIP_VINBIG.exists():
    print("\nEstrazione VinBigData in corso...")
    with zipfile.ZipFile(str(ZIP_VINBIG), 'r') as archive:
        file_nel_zip = set(archive.namelist())
        if 'train.csv' in file_nel_zip:
            with archive.open('train.csv') as f:
                df_vin = pd.read_csv(f)
            vin_perfect_ids = df_vin[df_vin['class_id'] == 14]['image_id'].unique()[:1200]
            
            for img_id in tqdm(vin_perfect_ids, desc="VinBigData"):
                internal_path = f"train/{img_id}.dicom"
                if internal_path in file_nel_zip:
                    try:
                        img_data = archive.read(internal_path)
                        dicom_file = pydicom.dcmread(io.BytesIO(img_data))
                        img = dicom_file.pixel_array.astype(float)
                        img = ((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5) * 255).astype(np.uint8)
                        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                        dataset_rows.extend(process_perfect_image(img))
                    except Exception: pass

if ZIP_PAESAGGI.exists():
    print("\nEstrazione Paesaggi in corso...")
    with zipfile.ZipFile(str(ZIP_PAESAGGI), 'r') as archive:
        paesaggi = [f for f in archive.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:1500]
        for path in tqdm(paesaggi, desc="Paesaggi"):
            try:
                img = cv2.imdecode(np.frombuffer(archive.read(path), np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                    dataset_rows.append({**extract_features(img), 'class_label': 'Inutilizzabile'})
            except Exception: pass

if dataset_rows:
    final_df = pd.DataFrame(dataset_rows)
    final_df.to_csv(str(PATH_OUTPUT_CSV), index=False)
    print(f"\nVITTORIA! File CSV salvato nella tua Home: {PATH_OUTPUT_CSV}")