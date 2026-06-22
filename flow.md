# IMPLEMENTATION ROADMAP FOR AI AGENT

## Personalized Stress Detection using HSSL + DPBL on WESAD

---

# FASE 0 – INFRASTRUKTUR DAN PERSIAPAN EKSPERIMEN

## Tujuan

Membangun fondasi aplikasi yang reproducible dan fault-tolerant sebelum implementasi model.

## Tasks

### Project Structure

* Membuat struktur folder standar:

  * config/
  * data/raw/
  * data/processed/
  * data/normalized/
  * checkpoints/
  * embeddings/
  * logs/
  * reports/
  * results/
  * src/
  * experiments/

### Configuration Management

* Membuat config.yaml.
* Menyimpan seluruh hyperparameter.
* Menghilangkan hardcoded path.
* Menetapkan random seed.

### Logging System

* Logging ke file dan console.
* Timestamp otomatis.
* Level log: INFO, WARNING, ERROR.

### Experiment Tracking

* Menyimpan metadata eksperimen.
* Menyimpan konfigurasi eksperimen.
* Menyimpan versi aplikasi.

## Deliverables

* config.yaml
* logger.py
* experiment_tracker.py

## Exit Criteria

* Struktur folder tervalidasi.
* Config dapat dibaca.
* Logging berjalan.

---

# FASE 1 – DATA ACQUISITION DAN PREPROCESSING

## Tujuan

Mempersiapkan dataset WESAD menjadi dataset siap eksperimen.

## Tasks

### Dataset Acquisition

* Download dataset WESAD.
* Skip download jika dataset sudah tersedia.
* Validasi struktur folder dataset.
* Validasi jumlah subject.

### Dataset Validation

* Verifikasi file sensor.
* Verifikasi label.
* Deteksi file corrupt.
* Analisis distribusi label.

### Signal Preprocessing

* Filtering noise.
* Missing value handling.
* Sinkronisasi sinyal.

### Windowing

* Configurable window size.
* Configurable overlap.
* Validasi jumlah window.
* Validasi label window.

### Normalization

* Fit scaler hanya pada training set.
* Transform training dan testing.
* Simpan scaler.

### Checkpoint

* Simpan hasil preprocessing.
* Skip preprocessing jika checkpoint tersedia.

## Deliverables

* processed_dataset/
* normalized_dataset/
* dataset_validation.json
* scaler.pkl

## Exit Criteria

* Tidak ada NaN.
* Tidak ada Infinite.
* Distribusi label tervalidasi.
* Dataset siap digunakan seluruh model.

---

# FASE 2 – PERSIAPAN EKSPERIMEN LOSO

## Tujuan

Membangun skema evaluasi antar subjek.

## Tasks

### LOSO Generator

* Generate seluruh fold LOSO.
* Satu subject sebagai test.
* Subject lainnya sebagai training.

### Validation

* Pastikan tidak ada data leakage.
* Subject test tidak muncul pada training.

### Export

* Simpan konfigurasi fold.

## Deliverables

* loso_folds.json

## Exit Criteria

* Seluruh fold berhasil dibuat.
* Data leakage tidak ditemukan.

---

# FASE 3 – PENGEMBANGAN MODEL HSSL

## Tujuan

Membangun representasi fitur hierarkis dari biosignal.

## Tasks

### Data Augmentation

* Implementasi augmentasi SSL.
* Validasi output augmentasi.

### HSSL Encoder

* Implementasi encoder hierarkis.
* Ekstraksi representasi mikro.
* Ekstraksi representasi makro.

### Contrastive Learning

* Implementasi objective SSL.
* Monitoring contrastive loss.

### Embedding Generation

* Generate embedding seluruh fold.
* Simpan embedding.

### Checkpoint

* Save best encoder.
* Save latest encoder.
* Resume training jika terputus.

## Deliverables

* embeddings/hssl/
* encoder checkpoints

## Exit Criteria

* Loss stabil.
* Tidak terjadi collapse.
* Embedding tervalidasi.

---

# FASE 4 – PERSONALIZATION MENGGUNAKAN DPBL

## Tujuan

Melakukan adaptasi representasi HSSL menjadi personal.

## Tasks

### DPBL Initialization

* Load embedding HSSL.
* Inisialisasi baseline personal.

### Baseline Update

* Update baseline tiap subject.
* Monitoring perubahan baseline.

### Deviation Computation

* Hitung deviasi terhadap baseline personal.

### Personalized Representation

* Generate representasi personal.

### Checkpoint

* Simpan model DPBL.
* Resume jika training gagal.

## Deliverables

* embeddings/hssl_dpbl/
* dpbl checkpoints

## Exit Criteria

* Adaptasi personal berjalan.
* Representasi personal berhasil dibuat.

---

# FASE 5 – CLASSIFIER DAN BASELINE MODEL

## Tujuan

Melatih seluruh model pembanding.

## Tasks

### Random Forest

* Training.
* Evaluation.
* Save model.

### 1D-CNN

* Training.
* Evaluation.
* Save model.

### SSL

* Training.
* Evaluation.
* Save model.

### HSSL

* Training.
* Evaluation.
* Save model.

### HSSL + DPBL

* Training classifier.
* Evaluation.
* Save model.

## Deliverables

* RF model
* CNN model
* SSL model
* HSSL model
* HSSL+DPBL model

## Exit Criteria

* Seluruh model menghasilkan prediksi.

---

# FASE 6 – EVALUASI MODEL

## Tujuan

Membandingkan performa seluruh model.

## Tasks

### LOSO Evaluation

* Evaluasi setiap fold.

### Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

### Export

* Simpan hasil tiap fold.

## Deliverables

* fold_results.csv

## Exit Criteria

* Seluruh fold memiliki metric lengkap.

---

# FASE 7 – ROBUSTNESS TESTING

## Tujuan

Memastikan kestabilan model.

## Tasks

### Repeated Experiment

* Jalankan eksperimen sebanyak 30 kali.
* Gunakan seed berbeda.

### Statistics

* Mean
* Standard Deviation
* Minimum
* Maximum

### Export

* Simpan hasil robustness.

## Deliverables

* robustness_results.csv

## Exit Criteria

* Seluruh 30 eksperimen selesai.

---

# FASE 8 – VALIDASI STATISTIK

## Tujuan

Membuktikan signifikansi hasil penelitian.

## Tasks

### Normality Test

* Shapiro-Wilk.

### Multi-model Comparison

* ANOVA.
* Jika asumsi tidak terpenuhi → Friedman Test.

### Post-hoc Analysis

* Tukey HSD.

### Contribution Analysis

* Paired t-Test:

  * HSSL vs HSSL+DPBL.

### Non-parametric Validation

* Wilcoxon Signed-Rank:

  * HSSL vs HSSL+DPBL.

### Export

* Simpan seluruh p-value.

## Deliverables

* statistical_results.csv

## Exit Criteria

* Seluruh uji statistik selesai.

---

# FASE 9 – REPORTING DAN VISUALISASI

## Tujuan

Menghasilkan artefak akhir penelitian.

## Tasks

### Tables

* Tabel performa seluruh model.
* Tabel robustness.
* Tabel hasil statistik.

### Visualization

* ROC Curve.
* PR Curve.
* Confusion Matrix.
* Boxplot perbandingan model.

### Final Report

* Generate laporan otomatis.

## Deliverables

* figures/
* final_tables/
* final_report/

## Exit Criteria

* Seluruh hasil siap digunakan untuk publikasi dan sidang.

---

# GLOBAL REQUIREMENTS

* Seluruh modul wajib memiliki logging.
* Seluruh modul wajib memiliki error handling.
* Seluruh proses wajib memiliki checkpoint.
* Seluruh proses wajib dapat di-resume.
* Skip proses jika output checkpoint sudah tersedia.
* Tidak boleh terjadi data leakage.
* Seluruh eksperimen harus reproducible.
* Semua output harus tersimpan dalam format CSV/JSON.
