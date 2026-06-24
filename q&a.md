# AUDIT IMPLEMENTASI PENELITIAN

## A. WORKFLOW PENELITIAN

1. **Sesuai sebagian.**

- **Beda:** Baseline "SSL" murni tidak ada (langsung RF/CNN → HSSL). Training RF/CNN paralel dengan HSSL, bukan sekuensial.
- **File:** `run_pipeline.py`.
- **Fungsi:** `run_command()`.
- **Bukti:** Line 40 (`models rf,cnn`), Line 45 (`models hssl`), tidak ada `ssl`.

## B. DATASET DAN PREPROCESSING

2. **Pemrosesan WESAD:**

- **Filtering:** Sinyal Chest EDA. Label 1,2,3 diambil. Label 2 → 1 (stress), lainnya → 0. (`src/data_preprocessing.py`, line 42-49).
- **Windowing:** Size 700, Overlap 0.5. (`src/generate_windows.py`, line 12). _Catatan: `config.yaml` menulis 60 (tidak konsisten)._
- **Normalisasi:** StandardScaler. (`src/loso_preparation.py`, line 59).

3. **Checkpoint Mechanism:**

- **Lokasi:** `data/processed/`, `embeddings/`, `results/`.
- **Mekanisme Skip:** `if os.path.exists(out_path): continue`. (`src/data_preprocessing.py` line 27; `src/run_all_folds.py` line 32 via arg `--skip_existing`).

## C. LOSO DAN DATA LEAKAGE

4. **LOSO Benar:**

- **Leakage Test:** Test subject tidak masuk train. (`src/loso_preparation.py`, line 44: `if test_subject in train_subjects: error`).

5. **Normalisasi Benar:**

- **Train Only Fit:** `scaler.fit(np.vstack(train_features))`. (`src/loso_preparation.py`, line 59).
- **Transform Semua:** `scaler.transform(data["features"])`. (`src/loso_preparation.py`, line 75).

## D. MODEL YANG DIEVALUASI

6. **Implementasi Aktual:**

- **Random Forest:** `src/train_classifiers.py` (`train_rf()`). Output: `rf_fold_*.json`.
- **1D-CNN:** `src/train_classifiers.py` (`train_cnn()`). Class: `Simple1DCNN`. Output: `cnn_fold_*.json`.
- **SSL:** **BELUM ADA IMPLEMENTASI.**
- **HSSL:** `src/train_hssl.py`, `src/models/hssl.py` (`HSSLEncoder`). Output: `hssl_fold_*.json`.
- **HSSL+DPBL:** `src/train_dpbl.py`, `src/models/dpbl.py` (`DPBL`). Output: `hssl+dpbl_fold_*.json`.

## E. HSSL

7. **SSL Approach Benar:**

- **Augmentasi:** Add Noise & Scale. (`src/augmentations.py`, line 14).
- **Loss:** NT-Xent Loss temperature 0.5. (`src/models/ssl_loss.py`, line 6).
- **Encoder:** `HSSLEncoder` (Micro 1D Conv + Macro Dilated Conv). (`src/models/hssl.py`, line 4).
- **Checkpoint:** `checkpoints/hssl_fold_{subj}/best.pt`. (`src/train_hssl.py`, line 127).

8. **Embedding Tersimpan:**

- **Lokasi:** `embeddings/hssl/fold_{subj}/*_embeddings.pkl`.
- **Format:** `.pkl` berisi dict `{"micro", "macro", "labels"}`.
- **Ukuran:** Sesuai `hidden_dim * 4` encoder. (`src/generate_embeddings.py`, line 70).

## F. DPBL

9. **DPBL Implementasi:**

- **Baseline:** Mean dari embedding label baseline (atau 50 window pertama). (`src/models/dpbl.py`, line 54 `BaselineTracker`).
- **Deviation:** `embeddings - baseline`. (`src/models/dpbl.py`, line 29).
- **Personalized:** Sigmoid Gating dari generic + projected deviation. (`src/models/dpbl.py`, line 36).

10. **Embedding Personal:** Ya. Dihasilkan via model `DPBL` pada embeddings HSSL. (`src/train_dpbl.py` dan disinggung di `run_pipeline.py` line 52).

## G. EVALUASI MODEL

11. **Metrik:** Lengkap (Acc, Prec, Rec, F1, ROC-AUC, PR-AUC, Conf Matrix). (`src/train_classifiers.py`, line 84).
12. **Fold LOSO:** Ya. Contoh: `results/rf_fold_S2.json`, dll. (`src/train_classifiers.py`, line 120).
13. **Phase 6 Output:** `fold_results.csv`, dll disinggung di config `evaluation`, dieksekusi via `evaluate_models.py` di pipeline.

## H. FOKUS PENELITIAN (HSSL+DPBL)

14. **Fokus Tercapai:** HSSL+DPBL dievaluasi (`models hssl+dpbl` di pipeline).
15. **Analisis Langsung:** Ya. Fungsi `pairwise_contribution` menghitung spesifik selisih mean dan p-value Wilcoxon/T-Test antara HSSL dan HSSL+DPBL. (`src/statistical_validation.py`, line 36).

## I. ROBUSTNESS TESTING

16. **Sesuai Desain:** Ya, di `src/robustness_testing.py` (via pipeline).
17. **Iterasi:** Default `--robust_iter 30`. (`run_pipeline.py`, line 21).
18. **Simpan:** Ya (struktur iterasi di `results/`).
19. **Statistik:** Ya, bagian dari output iteration dataframe.

## J. VALIDASI STATISTIK

20. **Phase 8:** Sudah lengkap (sebagian). File `src/statistical_validation.py`.
21. **Ketersediaan:**

- Ada: Shapiro-Wilk, Tukey HSD, Dunn, Paired t-Test, Wilcoxon.
- **Belum Ada:** ANOVA, Friedman Test.

## K. DASHBOARD DAN REPORTING

22. **Dashboard:** Mampu menjawab metrik dan perbandingan (via visualisasi di `generate_dashboard.py`).
23. **Visualisasi:** Disiapkan via `reporting_visualization.py` (tereksekusi di pipeline akhir).

## L. OUTPUT AKHIR

24. **Struktur:**
    `checkpoints/`, `data/processed/`, `data/normalized/`, `data/windowed/`, `embeddings/hssl/`, `embeddings/hssl_dpbl/`, `reports/`, `results/`.

## M. PERTANYAAN PALING PENTING

25. **Sesuai Tujuan:** Hampir selesai. Alur utama dan komparasi bekerja.
26. **Belum Sesuai Desain:**

- Model Baseline murni "SSL" tidak ada.
- Tidak ada Friedman / ANOVA test di validasi statistik.

27. **5 Kekurangan Terbesar:**
1. **SSL Baseline Hilang** (hanya ada HSSL).
1. **Potensi Data Leakage DPBL Baseline:** Penggunaan `np.where(labels == 1)` pada test fold. Baseline test harusnya mutlak diambil dari X menit pertama sesi fisik, BUKAN di-cheat via ground truth WESAD.
1. **Inkonsistensi Config vs Code:** `config.yaml` window=60, kode=700.
1. **Validasi Statistik:** Kurang uji omnibus (ANOVA/Friedman) sebelum Post-Hoc.
1. **CNN Hardcode Shape:** Transpose dim 2/3 sedikit brittle jika input channel berubah.
1. **Bisa Dipertanggungjawabkan?** **Belum.** Data leakage (poin 27.2) pada baseline DPBL untuk test subject akan merusak validitas sidang. Harus pakai _first N windows_ tanpa melihat label untuk test subject.
1. **Cukup Membuktikan HSSL+DPBL > Baseline?** Belum. "SSL" baseline tidak ada untuk membuktikan bahwa struktur hierarkis (H) memberikan efek.
1. **Robustness Cukup?** Ya, loop seed x30 sudah kuat.

**Kaggle Ready?** Ya. Semua `argparse` didukung. Pastikan `--skip_existing` dihapus / di-reset di Kaggle, dan struktur folder dibuat saat script run (via `os.makedirs`).

============== JAWABAN ke 2 =================

# AUDIT IMPLEMENTASI HSSL+DPBL — LAPORAN LENGKAP

## A. WORKFLOW PENELITIAN (Q1)

**Status: ✅ SESUAI** — Pipeline (`run_pipeline.py`) mengikuti alur yang benar:

| Tahap                  | File                                                    | Line                         | Status |
| ---------------------- | ------------------------------------------------------- | ---------------------------- | ------ |
| Data Acquisition       | `src/data_acquisition.py`                               | run_pipeline:29              | ✅     |
| Preprocessing          | `src/data_preprocessing.py`                             | run_pipeline:31              | ✅     |
| LOSO Split             | `src/loso_preparation.py`                               | run_pipeline:33              | ✅     |
| Window Generation      | `src/generate_windows.py`                               | run_pipeline:35              | ✅     |
| SSL Baseline           | `src/train_ssl.py`                                      | — **TIDAK ADA di pipeline!** | ❌     |
| HSSL Embedding         | `src/train_hssl.py` → `src/generate_embeddings.py`      | run_pipeline:43-44           | ✅     |
| DPBL Personalization   | `src/train_dpbl.py` → `src/generate_dpbl_embeddings.py` | run_pipeline:50-51           | ✅     |
| Classifier             | `src/train_classifiers.py` via `run_all_folds.py`       | run_pipeline:39,45,52        | ✅     |
| Evaluation             | `src/evaluate_models.py`                                | run_pipeline:62              | ✅     |
| Robustness Testing     | `src/robustness_testing.py`                             | run_pipeline:57              | ✅     |
| Statistical Validation | `src/statistical_validation.py`                         | run_pipeline:63              | ✅     |
| Dashboard              | `src/generate_dashboard.py`                             | run_pipeline:65              | ✅     |

**ISU KRITIS:** SSL pre-training (`train_ssl.py`) TIDAK dipanggil di pipeline mode `all` maupun mode `server_b`. Hanya HSSL yang dilatih. SSL embeddings juga tidak di-generate di pipeline. Akibatnya:

- Model SSL dan SSL+DPBL tidak akan memiliki embeddings
- Evaluasi akan gagal/missing untuk model SSL dan SSL+DPBL
- Perbandingan ablation tidak lengkap

---

## B. DATASET & PREPROCESSING (Q2-Q3)

### Q2. Filtering/Windowing/Normalisasi

**Filtering:**

- `src/data_preprocessing.py:47-49` — Hanya label [1,2,3] (baseline, stress, amusement) → binarize: stress=1, lainnya=0
- **Hanya chest EDA** (`data_preprocessing.py:42-43`) — BVP, ACC, TEMP tidak digunakan

**Windowing:**

- `src/generate_windows.py` — window_size=60 (config), overlap=0.5
- `src/train_hssl.py:20-21` — WESADDataset window_size=700, overlap=0.5 **(inkonsisten! config 60 vs hardcode 700)**

**Normalisasi:**

- `src/loso_preparation.py:61-62` — StandardScaler fit di training subjects
- `src/loso_preparation.py:87` — transform di train + test

### Q3. Checkpoint/Resume

✅ Checkpoint ada:

- `data_preprocessing.py:28-30` — skip if \_processed.pkl exists
- `generate_windows.py` — skip if \_windows.pkl exists
- `train_hssl.py:126-133` — resume training dari latest.pt
- `robustness_testing.py:65-70` — skip fold jika hasil sudah ada

---

## C. LOSO & DATA LEAKAGE (Q4-Q5)

### Q4. LOSO Correctness

✅ **Correct:**

- `loso_preparation.py:44-50` — train = semua subject kecuali test_subject
- `loso_preparation.py:48-50` — **explicit data leakage check**

### Q5. Normalisasi Leakage

✅ **No leakage:**

- `loso_preparation.py:61` — `scaler.fit(np.vstack(train_features))` **hanya pada training data**
- `loso_preparation.py:87` — test data hanya `.transform()`
- Scaler disimpan per fold: `checkpoints/scalers/scaler_fold_{subject}.pkl`

---

## D. MODEL YANG DIEVALUASI (Q6)

| Model         | File Implementasi                                                     | Class                                          | Status                      |
| ------------- | --------------------------------------------------------------------- | ---------------------------------------------- | --------------------------- |
| Random Forest | `train_classifiers.py:161-182`                                        | `ClassifierTrainer.train_rf()`                 | ✅                          |
| 1D-CNN        | `models/cnn.py`, `train_classifiers.py:184-255`                       | `Simple1DCNN`, `ClassifierTrainer.train_cnn()` | ✅                          |
| SSL           | `models/ssl/simsiam.py`, `train_ssl.py`, `generate_ssl_embeddings.py` | `SSLEncoder`, `SSLTrainer`                     | ✅ (tapi tidak di-pipeline) |
| HSSL          | `models/hssl.py`, `train_hssl.py`, `generate_embeddings.py`           | `HSSLEncoder`, `HSSLTrainer`                   | ✅                          |
| HSSL+DPBL     | `models/dpbl.py`, `train_dpbl.py`, `generate_dpbl_embeddings.py`      | `DPBL`, `BaselineTracker`                      | ✅                          |

**Output per model:** `results/{model}_fold_{subject}.json` — berisi metrics per fold.

---

## E. HSSL (Q7-Q8)

### Q7. Self-Supervised Learning

✅ **Ya:**

- **Augmentasi:** `src/augmentations.py:15-21` — noise + scaling (2 views berbeda)
- **Loss function:** `src/models/ssl_loss.py` — NTXentLoss (contrastive)
- **Encoder:** `src/models/hssl.py:4-58` — HSSLEncoder (micro + macro + projection head)
- **Checkpoint:** `checkpoints/hssl_fold_{subject}/latest.pt` dan `best.pt`

**ISU:** Augmentasi hanya noise + scale. Tidak ada augmentasi sinyal lain seperti time warp, channel shuffle, magnitude warp.

### Q8. Embedding Storage

✅ **Ya:**

- Lokasi: `embeddings/hssl/fold_{subject}/{subj}_embeddings.pkl`
- Format: pickle dict dengan keys `micro`, `macro`, `labels`
- HSSLEncoder output: micro_h=(Batch, 128, T), macro_h=(Batch, 128), z=(Batch, 128)
- Jumlah: 1 file per subject per fold

---

## F. DPBL (Q9-Q10)

### Q9. Implementasi DPBL

`src/models/dpbl.py`:

- **Personal baseline:** `BaselineTracker.update_baseline()` (line 60-74) — mean embeddings per subject
- **Deviation:** `DPBL.forward()` line 34 — `deviation = embeddings - baseline`
- **Personalized representation:** line 37-45 — concat → projection → gating → weighted sum
- **Training:** `src/train_dpbl.py` — train DPBL module with contrastive loss
- **Generate embeddings:** `src/generate_dpbl_embeddings.py` — load HSSL embeddings → DPBL forward → save

### Q10. Input/Output

✅ **Input:** HSSL macro embeddings (dari `embeddings/hssl/`)
✅ **Output:** personalized embeddings (`embeddings/hssl_dpbl/`) — key `macro_dpbl`

---

## G. EVALUASI MODEL (Q11-Q13)

### Q11. Metrics

✅ **Semua dihitung:**

- `evaluate_models.py:76-83` — accuracy, precision, recall, f1_score (macro + weighted)
- `evaluate_models.py:91-108` — roc_auc, pr_auc
- `evaluate_models.py:86-88` — confusion_matrix

### Q12. Per-fold JSON

✅ File output: `results/{model}_fold_{subject}.json` — 1 file per fold per model.

**ISU:** Di `evaluate_models.py:23`, model list mencakup `"ssl+dpbl"` dan `"hssl+dpbl"`, TAPI di fold_results.csv nama model `"ssl+dpbl"` dan `"hssl+dpbl"` disimpan dengan format seperti itu. Sementara di `train_classifiers.py`, nama model untuk training adalah `"SSL+DPBL"` dan `"HSSL+DPBL"`. **Inkonsistensi kapitalisasi bisa menyebabkan missing evaluation.**

### Q13. Phase 6 Output

✅ Semua output dihasilkan:

- `results/fold_results.csv`
- `results/model_summary.csv`
- `results/confusion_matrices/{model}_confusion_matrix.json`
- `results/evaluation_report.json`
- `results/evaluation_metadata.json`

---

## H. FOKUS PENELITIAN (Q14-Q15)

### Q14. HSSL+DPBL sebagai Fokus

⚠️ **SEBAGIAN:** HSSL+DPBL adalah fokus arsitektur, tapi:

- Robustness testing default n_iters=3, bukan 30 (desain)
- SSL pre-training tidak jalan di pipeline → ablation tidak lengkap
- **Tidak ada analisis HSSL vs HSSL+DPBL yang menghasilkan output spesifik**

### Q15. Analisis HSSL vs HSSL+DPBL

❌ **TIDAK ADA ANALISIS DEDIKASI:**

- `statistical_validation.py:188-216` — ada `pairwise_contribution()` tapi ini membaca dari robustness iterations, bukan dari per-fold comparison
- Tidak ada skrip atau output yang langsung membandingkan per-subject: HSSL vs HSSL+DPBL

---

## I. ROBUSTNESS TESTING (Q16-Q19)

### Q16. Desain vs Implementasi

| Aspek           | Desain | Implementasi                                              | Status                      |
| --------------- | ------ | --------------------------------------------------------- | --------------------------- |
| Hanya HSSL+DPBL | ✅     | `robustness_testing.py:28`                                | ✅                          |
| 30 iterasi      | ✅     | Default `--n_iters=3` (argumen)                           | ⚠️ **default tidak sesuai** |
| Seed berbeda    | ✅     | `robustness_testing.py:213` — random seeds dari base_seed | ✅                          |

### Q17. Default Iterasi

❌ **Default 3** (`robustness_testing.py:254`). Desain penelitian 30.

### Q18. Output Per Iterasi

✅ Setiap iterasi disimpan di: `results/robustness/iter_{idx}_seed_{seed}/`

### Q19. Statistik Robustness

✅ `robustness_testing.py:153-157` — mean, std, min, max, CI 95%
✅ `robustness_testing.py:168-189` — export ke `results/robustness/robustness_results.csv`

---

## J. VALIDASI STATISTIK (Q20-Q21)

### Q20. Implementasi

✅ **SUDAH LENGKAP** — `src/statistical_validation.py`:

### Q21. Ketersediaan Uji

| Uji                  | Line                            | Status |
| -------------------- | ------------------------------- | ------ |
| Shapiro-Wilk         | `statistical_validation.py:99`  | ✅     |
| ANOVA                | `statistical_validation.py:124` | ✅     |
| Friedman Test        | `statistical_validation.py:136` | ✅     |
| Paired t-Test        | `statistical_validation.py:200` | ✅     |
| Wilcoxon Signed-Rank | `statistical_validation.py:201` | ✅     |
| Tukey HSD            | `statistical_validation.py:157` | ✅     |
| Dunn's Test          | `statistical_validation.py:162` | ✅     |
| Cohen's d            | `statistical_validation.py:35`  | ✅     |
| Hedges' g            | `statistical_validation.py:43`  | ✅     |

**ISU:** Statistical validation bergantung pada `results/robustness/` yang berisi iterasi robustness. Jika robustness belum dijalankan (n_iters=0 atau folder kosong), validasi statistik akan return empty.

---

## K. DASHBOARD (Q22-Q23)

### Q22. Pertanyaan yang Dapat Dijawab

| Pertanyaan              | Status     | Keterangan                           |
| ----------------------- | ---------- | ------------------------------------ |
| Model terbaik?          | ⚠️ Parsial | Tergantung data di model_summary.csv |
| Peningkatan HSSL+DPBL?  | ⚠️ Parsial | Butuh perbandingan dedicated         |
| Subject paling sulit?   | ❌ Belum   | Tidak ada analisis per-subject       |
| Stabilitas robustness?  | ✅ Bisa    | Dari robustness_results.csv          |
| Signifikansi statistik? | ⚠️ Parsial | Jika robustness dijalankan           |

### Q23. Visualisasi

| Visualisasi          | Status                          |
| -------------------- | ------------------------------- |
| Benchmark comparison | ✅ `reporting_visualization.py` |
| ROC Curve            | ✅                              |
| PR Curve             | ✅                              |
| Confusion Matrix     | ✅                              |
| Robustness Plot      | ✅                              |
| Statistical Summary  | ✅                              |

---

## L. OUTPUT FOLDER (Q24)

```
results/
  fold_results.csv
  model_summary.csv
  evaluation_report.json
  evaluation_metadata.json
  predictions/            # {model}_fold_{subject}.json
  confusion_matrices/     # {model}_confusion_matrix.json
  learning_curves/        # cnn_fold_{subject}.json
  robustness/
    robustness_results.csv
    robustness_metadata.json
    iter_1_seed_*/        # per iteration results
  statistical_analysis/
    normality_results.csv
    multimodel_comparison.json
    posthoc_results.csv
    effect_sizes.csv
    contribution_analysis.json
reports/
  dataset_validation.json
  loso_folds.json
checkpoints/
  scalers/
  ssl_pretrain/
  hssl_fold_{subject}/
  {model}_fold_{subject}.pt
  rf_fold_{subject}.pkl
embeddings/
  hssl/fold_{subject}/
  hssl_dpbl/fold_{subject}/
  ssl/fold_{subject}/
  ssl_dpbl/fold_{subject}/
dashboard/
  interactive_dashboard.html
```

---

## M. PERTANYAAN PALING KRITIS (Q25-Q30)

### Q25. Apakah sesuai tujuan penelitian?

**⚠️ SEBAGIAN.** Arsitektur sudah benar, pipeline lengkap. Tapi ada beberapa gap.

### Q26-Q27. 5 KEKURANGAN TERBESAR

1. **❌ SSL pre-training tidak berjalan di pipeline** → Semua model SSL dan SSL+DPBL akan missing. Ablation tidak lengkap.
   - Fix: Tambahkan langkah SSL di `run_pipeline.py` (server_a atau server_b)

2. **❌ Robustness testing default n_iters=3, desain 30** → Hasil tidak cukup untuk klaim stabilitas ilmiah.
   - Fix: Ubah default di `robustness_testing.py:254` dari 3 ke 30; di `run_pipeline.py:21` dari 30 ke 30 (sudah benar)

3. **❌ Inkonsistensi window_size** — Config 60, HSSL hardcode 700.
   - `loso_preparation.py` dan train_classifiers menggunakan config (60)
   - `train_hssl.py` hardcode 700 (line 81)
   - `generate_embeddings.py` hardcode 700 (line 21)
   - `train_ssl.py` hardcode 700

4. **❌ Tidak ada analisis per-subject** → Tidak bisa menjawab "subject mana yang paling sulit" tanpa analisis dedicated.

5. **❌ Preprocessing hanya EDA** — WESAD punya BVP, ACC, TEMP. Single channel sangat membatasi potensi HSSL dengan multi-channel representations.

### Q28. Apakah hasil dapat dipertanggungjawabkan?

**⚠️ BELUM SEPENUHNYA.**

- LOSO benar ✅
- No data leakage ✅
- Metrics lengkap ✅
- Tapi: SSL baseline missing, window_size inkonsisten, preprocessing hanya EDA, robustness default 3 iterasi
- Konsekuensi: Paper bisa direview dengan pertanyaan "why only EDA?", "why inconsistent window sizes?", "why SSL baseline missing?"

### Q29. Apakah cukup membuktikan HSSL+DPBL lebih baik?

**❌ BELUM.**

- Tidak ada perbandingan langsung HSSL vs HSSL+DPBL per subject
- SSL dan SSL+DPBL tidak jalan → tidak bisa klaim "HSSL+DPBL better than SSL+DPBL"
- Butuh dedicated analysis script untuk pairwise comparison

### Q30. Apakah cukup membuktikan stabilitas?

**⚠️ Parsial.**

- Robustness framework benar ✅
- Tapi default 3 iterasi tidak cukup → perlu 30
- Statistical validation sudah benar ✅ jika data tersedia

### KESIAPAN KAGGLE

**Konfigurasi untuk Kaggle:**

1. **Path:** Ubah `config/config.yaml` — points ke `/kaggle/input/wesad-dataset` untuk raw data, dan `/kaggle/working/` untuk output
2. **Dependencies:** `requirements.txt` sudah cukup baik. Tambahkan: `scikit-posthocs` (untuk Dunn's test)
3. **Kaggle Notebook template:**

```python
# Kaggle Notebook — Cell 1: Setup
!pip install -r /kaggle/working/requirements.txt

# Cell 2: Download dataset (via Kaggle dataset API atau upload)
# WESAD dataset dari Kaggle: https://www.kaggle.com/datasets/...wesad

# Cell 3: Run full pipeline
!python run_pipeline.py --epochs 30 --ssl_epochs 100 --robust_iter 30 --mode all

# Cell 4: Atau distributed (jika waktu terbatas)
# Sesi 1: !python run_pipeline.py --mode server_a
# Sesi 2: !python run_pipeline.py --mode server_b
# Sesi 3: !python run_pipeline.py --mode server_c
# Sesi 4: !python run_pipeline.py --mode server_d
# Sesi 5: !python run_pipeline.py --mode eval
```

4. **Waktu:** Full pipeline (30 iterasi robustness) membutuhkan ~8-12 jam di GPU Kaggle (T4/P100)

### KESIMPULAN

| Area               | Status                                       |
| ------------------ | -------------------------------------------- |
| A. Workflow        | ⚠️ SSL missing di pipeline                   |
| B. Preprocessing   | ✅ Checkpoint, ❌ Hanya EDA                  |
| C. LOSO            | ✅ Correct                                   |
| D. Model           | ✅ All implemented                           |
| E. HSSL            | ✅ Correct SSL approach                      |
| F. DPBL            | ✅ Architecture correct                      |
| G. Evaluation      | ✅ Metrics lengkap                           |
| H. HSSL+DPBL Focus | ❌ No dedicated comparison                   |
| I. Robustness      | ⚠️ Default 3, but framework correct          |
| J. Statistical     | ✅ Complete implementation                   |
| K. Dashboard       | ✅ Available                                 |
| Kaggle Ready       | ⚠️ Need config adjustment + SSL pipeline fix |

**Recommended immediate fixes before sidang:**

1. Tambah SSL training di pipeline (insert between data_prep and HSSL)
2. Fix default n_iters → 30
3. Konsistenkan window_size (semua pakai config)
4. Tambah multi-channel preprocessing (BVP + EDA minimal)
5. Buat analysis script: HSSL vs HSSL+DPBL per-subject comparison
