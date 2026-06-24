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
