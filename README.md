# WESAD Stress Detection - HSSL+DPBL

This repository contains the complete codebase for stress detection using the WESAD dataset, comparing baseline models (Random Forest, 1D-CNN) against Self-Supervised Learning (SSL) architectures: standard Hierarchical SSL (HSSL) and the proposed HSSL combined with Dirichlet Prototype-Based Learning (HSSL+DPBL).

## Requirements

Ensure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

## Directory Structure

* `data/`: Raw dataset, processed signals, and normalized folds.
* `src/`: Core python scripts for the pipeline.
* `checkpoints/`: Saved PyTorch model weights.
* `embeddings/`: Extracted embeddings from HSSL and DPBL.
* `results/`: Output JSONs, predictions, tables, charts, and the final HTML Dashboard.

## How to Run the Pipeline (End-to-End)

We provide a single entry point script to run the entire evaluation pipeline across all Leave-One-Subject-Out (LOSO) folds, evaluate the results, test robustness, run statistical significance, and finally generate an interactive HTML dashboard.

```bash
python run_pipeline.py
```

### What does `run_pipeline.py` do?
1. **Training Classifiers (`src.run_all_folds`)**: Iterates through all subjects (S2-S17), trains RF, 1D-CNN, HSSL, and HSSL+DPBL, and saves the probability predictions.
2. **Evaluation (`src.evaluate_models`)**: Calculates Precision, Recall, F1-Score, Accuracy, and Confusion Matrices across all folds. 
3. **Robustness Testing (`src.robustness_testing`)**: Retrains the best model (HSSL+DPBL) 30 times with different random seeds to verify stability (calculates 95% Confidence Interval).
4. **Statistical Validation (`src.statistical_validation`)**: Performs Shapiro-Wilk normality tests and Wilcoxon Signed-Rank tests to prove the statistical significance of HSSL+DPBL against baselines.
5. **Dashboard Generation (`src.generate_dashboard`)**: Compiles all results into `results/interactive_dashboard.html`.

## Viewing the Results

Open `results/interactive_dashboard.html` in any web browser to view:
* **KPIs**: Best F1-Score, Contribution of DPBL.
* **Performance**: Radar charts, Bar charts (Mean F1, Fold breakdown).
* **Confusion Matrix Heatmaps**: Where the model makes mistakes.
* **Robustness**: F1-Score stability across 30 iterations (with error bars).
* **Statistical Validation**: P-values confirming significance.

Baik, saya mengerti. Saya akan memodifikasi arsitektur eksekusi agar sangat fleksibel. 

Saya akan menambahkan parameter `--mode` pada `run_pipeline.py` dan parameter `--models` pada `src/run_all_folds.py`.

Nantinya Anda bisa menjalankan perintah seperti ini:

**1 Server (Jalan Semua seperti biasa):**
`python run_pipeline.py --mode all`

**Banyak Server (Distributed):**
*   **Server Utama (Data Prep):** `python run_pipeline.py --mode data_prep` (Lalu copy folder `data/` ke server lain).
*   **Server A (RF & CNN):** `python run_pipeline.py --mode server_a`
*   **Server B (HSSL):** `python run_pipeline.py --mode server_b`
*   **Server C (HSSL+DPBL):** `python run_pipeline.py --mode server_c`
*   **Server D (Robustness):** `python run_pipeline.py --mode server_d`
*   **Server Utama (Agregasi):** `python run_pipeline.py --mode eval` (Setelah semua hasil disatukan).

Saya sudah merancang solusinya. Tolong **toggle ke Act mode** agar saya bisa mengimplementasikan kode ini ke dalam file `run_pipeline.py`, `src/run_all_folds.py`, dan `src/train_classifiers.py`.

Pipeline updated for distributed execution -> `run_pipeline.py` now accepts `--mode`.

Execution Modes:
- `--mode all` -> Runs entirely on 1 server.
- `--mode data_prep` -> Runs prep steps (Langkah 1-4). Transfer `data/` to other servers after.
- `--mode server_a` -> Runs RF & CNN.
- `--mode server_b` -> Runs HSSL Pre-training, Embeddings & Classifier.
- `--mode server_c` -> Runs DPBL Training, Embeddings & Classifier (requires `checkpoints/` from B).
- `--mode server_d` -> Runs Robustness (requires `embeddings/` from C).
- `--mode eval` -> Runs evaluation, stats, and dashboard (gather all `results/` & `checkpoints/` first).

Modifications:
- `src/train_classifiers.py` -> Added `models_to_run` parameter.
- `src/run_all_folds.py` -> Added `--models` flag.
- `run_pipeline.py` -> Configured conditional blocks per server.

Berikut adalah estimasi dan penjelasan untuk penggunaan di Kaggle:

**1. Estimasi Waktu (Sangat Cukup dengan 30 Jam/Minggu)**
Total waktu untuk menjalankan *full pipeline* (15 Folds LOSO) diperkirakan memakan waktu sekitar **10 hingga 15 jam** menggunakan GPU gratis Kaggle (T4 x2 atau P100).
*   **Data Preprocessing & Windowing:** ~1-2 jam (bergantung pada RAM/CPU).
*   **Training & Evaluasi (HSSL, DPBL, CNN, RF):** ~30-45 menit per subjek/fold. Untuk 15 subjek = ~10 jam.
*   **Robustness & Agregasi:** ~1-2 jam.
*   *Catatan Penting:* Kaggle memiliki batas waktu **12 jam per sesi eksekusi non-stop**. Oleh karena itu, fitur `--mode` dan `--skip_existing` yang sudah kita buat tadi akan sangat berguna. Jika sesi Kaggle mati di jam ke-12, Anda bisa *run* ulang dan program akan melanjutkan dari fold yang belum selesai secara otomatis tanpa mengulang dari awal.

**2. Apakah Perlu Download Dataset?**
**TIDAK PERLU** jika Anda mengeksekusi kode ini *di dalam* Kaggle Notebook.
Anda cukup melakukan:
1. Buka Kaggle Notebook Anda.
2. Klik tombol **"Add Data"** di panel kanan.
3. Cari "WESAD" dan klik **Add**.
4. Dataset akan otomatis terhubung (ter-mount) ke dalam notebook Anda (biasanya di path `/kaggle/input/wesad/...`) dalam hitungan detik tanpa memotong kuota internet.

Nantinya, Anda hanya perlu mengarahkan `raw_data_dir` di dalam file `config.yaml` ke path `/kaggle/input/...` tersebut.

Semua pekerjaan modifikasi skrip sudah selesai 100%. Anda bisa mulai mengunggah file-file ini ke Kaggle dan menjalankan `run_pipeline.py`. Ada hal lain yang ingin dipersiapkan sebelum Anda jalankan di Kaggle?