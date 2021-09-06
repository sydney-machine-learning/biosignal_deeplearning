# biosignal_deeplearning
Processing bio-signals with deep learning

How to use this repository?
1. Put the Capnobase, MIMIC-II, sEMG datasets in the Datasets folder.
2. Create Directory ('Saved/{Dataset}/{Model-Name}') for all datasets (Capno, MIMIC-II, sEMG) and models (LSTM, CNN, etc)
3. In main.py, update the directory paths. Update batch size as per memory. In case of GPU: Use Batchsize=10000, TPU: 100000,
4. Run main.py

1 run takes close to 20 hrs to run on TPU (Includes training on all 7 models).
Comment appropriate sections (models) in main.py to reduce compuatation time.
