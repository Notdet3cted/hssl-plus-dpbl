import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
from src.logger import setup_logger
from src.experiment_tracker import ExperimentTracker

class DataPreprocessing:
    def __init__(self):
        self.logger = setup_logger("DataPreprocessing")
        self.tracker = ExperimentTracker()
        self.raw_dir = os.path.join(self.tracker.config["paths"]["raw_data"], "WESAD")
        self.processed_dir = self.tracker.config["paths"]["processed_data"]
        self.subjects = [f"S{i}" for i in range(2, 18) if i != 12]
        self.window_size = self.tracker.config.get("preprocessing", {}).get("window_size", 700)
        self.overlap = self.tracker.config.get("preprocessing", {}).get("overlap", 0.5)
        self.channels = self.tracker.config.get("preprocessing", {}).get("channels", ["EDA"])
        self.target_fps = self.tracker.config.get("preprocessing", {}).get("target_fps", 700)

    def process_all(self):
        self.logger.info("Starting data preprocessing...")
        self.logger.info(f"Channels: {self.channels}")
        for subject in self.subjects:
            self._process_subject(subject)
        self.logger.info("Preprocessing completed.")

    def _resample_to_target(self, signal, orig_fs, target_fs):
        """Resample a signal from orig_fs to target_fs."""
        if orig_fs == target_fs:
            return signal
        n_target = int(len(signal) * target_fs / orig_fs)
        return resample(signal, n_target)

    def _process_subject(self, subject):
        subj_path = os.path.join(self.raw_dir, subject, f"{subject}.pkl")
        out_path = os.path.join(self.processed_dir, f"{subject}_processed.pkl")

        if os.path.exists(out_path):
            self.logger.info(f"Processed data for {subject} already exists. Skipping.")
            return

        if not os.path.exists(subj_path):
            self.logger.error(f"Raw data for {subject} not found. Skipping.")
            return

        self.logger.info(f"Processing {subject}...")
        try:
            with open(subj_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')

            chest_data = data['signal']['chest']
            labels = data['label']

            # WESAD chest signals and their native sampling rates
            CHEST_FS = 700  # All chest signals sampled at 700Hz

            # Determine target sampling rate
            target_fs = self.target_fps if self.target_fps else CHEST_FS

            # WESAD labels: 1=baseline, 2=stress, 3=amusement
            mask = np.isin(labels, [1, 2, 3])
            filtered_labels = labels[mask]
            # Band‑pass filter parameters from config
            fp = self.tracker.config.get("preprocessing", {}).get("filter", {})
            low = fp.get("low_cut", 0.3)
            high = fp.get("high_cut", 45.0)
            order = fp.get("order", 4)
            fs = self.target_fps if self.target_fps else 700
            nyq = 0.5 * fs
            low_norm = low / nyq
            high_norm = high / nyq
            b, a = butter(order, [low_norm, high_norm], btype='band')
            
            binary_labels = np.where(filtered_labels == 2, 1, 0)
            # Extract and align multi-channel features
            channel_arrays = []
            for ch_name in self.channels:
                ch_upper = ch_name.upper()
                if ch_upper in chest_data:
                raw_signal = chest_data[ch_upper].flatten()
                # Apply band‑pass filter
                try:
                    filtered_signal = filtfilt(b, a, raw_signal)
                except Exception as e:
                    self.logger.warning(f"Filtering failed for {ch_upper}: {e}")
                    filtered_signal = raw_signal
                # Apply label mask
                masked_signal = filtered_signal[mask]
                # Resample if needed
                if target_fs != CHEST_FS:
                    masked_signal = self._resample_to_target(masked_signal, CHEST_FS, target_fs)
                channel_arrays.append(masked_signal)
                self.logger.info(f"  {subject} channel {ch_upper}: {masked_signal.shape[0]} samples")
                else:
                    self.logger.warning(f"  Channel {ch_upper} not found in chest data for {subject}. Skipping channel.")

            if not channel_arrays:
                self.logger.error(f"No valid channels for {subject}. Skipping.")
                return

            # Align lengths (all chest signals should be same length, but resample may differ by 1)
            min_len = min(arr.shape[0] for arr in channel_arrays)
            channel_arrays = [arr[:min_len] for arr in channel_arrays]

            # If labels were resampled too
            if target_fs != CHEST_FS:
                binary_labels = self._resample_labels(binary_labels, CHEST_FS, target_fs, min_len)
            else:
                binary_labels = binary_labels[:min_len]

            # Stack into (T, C) shape
            features = np.stack(channel_arrays, axis=1)  # (T, n_channels)

            processed_data = {
                "features": features,
                "labels": binary_labels,
                "channels": self.channels,
                "sampling_rate": target_fs
            }

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'wb') as f:
                pickle.dump(processed_data, f)

            self.logger.info(f"Saved processed data for {subject}: features {features.shape}, labels {binary_labels.shape}")
        except Exception as e:
            self.logger.error(f"Error processing {subject}: {e}")
            import traceback
            traceback.print_exc()

    def _resample_labels(self, labels, orig_fs, target_fs, target_len):
        """Resample labels by nearest-neighbor to match resampled signal length."""
        orig_len = len(labels)
        indices = np.round(np.linspace(0, orig_len - 1, target_len)).astype(int)
        return labels[indices]

if __name__ == "__main__":
    processor = DataPreprocessing()
    processor.process_all()