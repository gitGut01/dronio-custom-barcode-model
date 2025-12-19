30 min 30 epochs 20K dataset simple




2025-12-19 21:27:56.530609: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1766179676.562783    1678 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1766179676.572720    1678 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1766179676.597587    1678 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1766179676.597618    1678 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1766179676.597625    1678 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1766179676.597632    1678 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-12-19 21:27:56.604122: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
/usr/local/lib/python3.12/dist-packages/mlflow/tracking/_tracking_service/utils.py:177: FutureWarning: The filesystem tracking backend (e.g., './mlruns') will be deprecated in February 2026. Consider transitioning to a database backend (e.g., 'sqlite:///mlflow.db') to take advantage of the latest MLflow features. See https://github.com/mlflow/mlflow/issues/18534 for more details and migration guidance.
  return FileStore(store_uri, store_uri)
2025/12/19 21:28:11 INFO mlflow.tracking.fluent: Experiment with name 'barcode-ctc' does not exist. Creating a new experiment.
epoch=1 train_loss=2.5654 val_loss=2.4822 exact=0.000 CER=1.000 vocab=11
epoch=2 train_loss=2.3482 val_loss=2.1770 exact=0.000 CER=0.833 vocab=11
epoch=3 train_loss=1.9804 val_loss=1.7884 exact=0.009 CER=0.618 vocab=11
epoch=4 train_loss=1.6418 val_loss=1.5191 exact=0.053 CER=0.501 vocab=11
epoch=5 train_loss=1.4171 val_loss=1.3529 exact=0.083 CER=0.458 vocab=11
epoch=6 train_loss=1.2662 val_loss=1.2241 exact=0.130 CER=0.413 vocab=11
epoch=7 train_loss=1.1499 val_loss=1.1636 exact=0.144 CER=0.398 vocab=11
epoch=8 train_loss=1.0630 val_loss=1.0922 exact=0.160 CER=0.375 vocab=11
epoch=9 train_loss=0.9876 val_loss=1.0252 exact=0.195 CER=0.354 vocab=11
epoch=10 train_loss=0.9268 val_loss=0.9612 exact=0.199 CER=0.330 vocab=11
epoch=11 train_loss=0.8685 val_loss=0.9571 exact=0.196 CER=0.331 vocab=11
epoch=12 train_loss=0.8209 val_loss=0.9307 exact=0.207 CER=0.318 vocab=11
epoch=13 train_loss=0.7754 val_loss=0.8975 exact=0.216 CER=0.313 vocab=11
epoch=14 train_loss=0.7363 val_loss=0.9032 exact=0.233 CER=0.307 vocab=11
epoch=15 train_loss=0.6963 val_loss=0.8692 exact=0.233 CER=0.296 vocab=11
epoch=16 train_loss=0.6641 val_loss=0.8363 exact=0.244 CER=0.287 vocab=11
epoch=17 train_loss=0.6285 val_loss=0.8181 exact=0.247 CER=0.281 vocab=11
epoch=18 train_loss=0.5989 val_loss=0.8134 exact=0.270 CER=0.278 vocab=11
epoch=19 train_loss=0.5649 val_loss=0.8255 exact=0.254 CER=0.280 vocab=11
epoch=20 train_loss=0.5385 val_loss=0.8117 exact=0.262 CER=0.277 vocab=11
epoch=21 train_loss=0.5166 val_loss=0.7822 exact=0.280 CER=0.268 vocab=11
epoch=22 train_loss=0.4873 val_loss=0.8248 exact=0.268 CER=0.270 vocab=11
epoch=23 train_loss=0.4652 val_loss=0.8108 exact=0.256 CER=0.272 vocab=11
epoch=24 train_loss=0.4433 val_loss=0.7887 exact=0.279 CER=0.266 vocab=11
epoch=25 train_loss=0.4239 val_loss=0.7752 exact=0.281 CER=0.262 vocab=11
epoch=26 train_loss=0.3955 val_loss=0.8148 exact=0.281 CER=0.264 vocab=11
epoch=27 train_loss=0.3797 val_loss=0.7932 exact=0.286 CER=0.255 vocab=11
epoch=28 train_loss=0.3630 val_loss=0.8174 exact=0.290 CER=0.254 vocab=11
epoch=29 train_loss=0.3414 val_loss=0.8007 exact=0.278 CER=0.260 vocab=11
epoch=30 train_loss=0.3240 val_loss=0.8203 exact=0.296 CER=0.253 vocab=11
epoch=31 train_loss=0.3045 val_loss=0.8241 exact=0.289 CER=0.256 vocab=11
epoch=32 train_loss=0.2832 val_loss=0.8099 exact=0.273 CER=0.257 vocab=11