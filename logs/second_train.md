2025-12-20 09:17:09.839580: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-12-20 09:17:09.855982: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1766222229.876796    5983 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1766222229.883046    5983 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1766222229.898452    5983 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1766222229.898485    5983 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1766222229.898487    5983 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1766222229.898491    5983 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-12-20 09:17:09.903074: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
--- Loading train dataset ---
  > Processed 250000 rows...
  > Processed 500000 rows...
  > Processed 750000 rows...
  > Processed 1000000 rows...
Successfully loaded 1000000 samples for train.
--- Loading val dataset ---
Successfully loaded 100000 samples for val.
/usr/local/lib/python3.12/dist-packages/mlflow/tracking/_tracking_service/utils.py:177: FutureWarning: The filesystem tracking backend (e.g., './mlruns') will be deprecated in February 2026. Consider transitioning to a database backend (e.g., 'sqlite:///mlflow.db') to take advantage of the latest MLflow features. See https://github.com/mlflow/mlflow/issues/18534 for more details and migration guidance.
  return FileStore(store_uri, store_uri)
2025/12/20 09:17:33 INFO mlflow.tracking.fluent: Experiment with name 'barcode-ctc' does not exist. Creating a new experiment.
Epoch 1 [0/977] Loss: 4.3704
Epoch 1 [100/977] Loss: 2.5766
Epoch 1 [200/977] Loss: 2.5307
Epoch 1 [300/977] Loss: 2.4865
Epoch 1 [400/977] Loss: 2.3365
Epoch 1 [500/977] Loss: 1.9844
Epoch 1 [600/977] Loss: 1.5892
Epoch 1 [700/977] Loss: 1.2520
Epoch 1 [800/977] Loss: 1.1136
Epoch 1 [900/977] Loss: 0.9924
--- Epoch 1 Summary | Val Loss: 0.9853 | Acc: 20.03% ---
Epoch 2 [0/977] Loss: 0.9077
Epoch 2 [100/977] Loss: 0.8496
Epoch 2 [200/977] Loss: 0.7535
Epoch 2 [300/977] Loss: 0.7559
Epoch 2 [400/977] Loss: 0.6756
Epoch 2 [500/977] Loss: 0.6376
Epoch 2 [600/977] Loss: 0.5236
Epoch 2 [700/977] Loss: 0.5321
Epoch 2 [800/977] Loss: 0.4619
Epoch 2 [900/977] Loss: 0.4194
--- Epoch 2 Summary | Val Loss: 0.4261 | Acc: 49.77% ---
Epoch 3 [0/977] Loss: 0.3814
Epoch 3 [100/977] Loss: 0.3568
Epoch 3 [200/977] Loss: 0.3590
Epoch 3 [300/977] Loss: 0.3512
Epoch 3 [400/977] Loss: 0.3285
Epoch 3 [500/977] Loss: 0.3485
Epoch 3 [600/977] Loss: 0.2545
Epoch 3 [700/977] Loss: 0.2966
Epoch 3 [800/977] Loss: 0.2959
Epoch 3 [900/977] Loss: 0.2284
--- Epoch 3 Summary | Val Loss: 0.2448 | Acc: 64.10% ---
Epoch 4 [0/977] Loss: 0.2345
Epoch 4 [100/977] Loss: 0.2098