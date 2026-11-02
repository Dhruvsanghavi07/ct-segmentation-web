# Model card: spleen CT segmentation U-Net

## Intended use
Research and education: showing how a segmentation model is trained, evaluated per patient and deployed to run in a browser. **Not a medical device. Not for diagnosis, treatment planning or any clinical decision.**

## Model
2D U-Net (4 levels, 16 to 128 channels, 256-channel bottleneck), trained from scratch with cross-entropy plus soft Dice loss, AdamW and a cosine learning-rate schedule. Input: one axial CT slice, soft-tissue window [-160, 240] HU, resized to 128×128 and scaled to [0, 1]. Output: background/spleen logits per pixel. Post-processing keeps the largest 3D connected component.

## Data
Medical Segmentation Decathlon, Task 09 (Spleen): 41 portal-venous-phase abdominal CT volumes with expert labels, from patients undergoing chemotherapy for liver metastases at Memorial Sloan Kettering Cancer Center. License CC-BY-SA 4.0 ([Antonelli et al., 2022](https://www.nature.com/articles/s41467-022-30695-9)). Split by patient with a fixed seed: 29 train, 6 validation, 6 test. The split is saved with every run (`split.json` in the release).

## Evaluation
Per-patient 3D Dice and 95th-percentile Hausdorff distance (HD95, mm) on the 6 held-out test patients, computed at the model's 128×128 in-plane resolution. Each run's numbers are in its GitHub Release and in the README.

## Limitations
- Six test patients is a small sample: the mean Dice has wide uncertainty, and one hard case moves it a lot.
- Single institution, single contrast phase. Performance on other scanners, protocols or non-contrast CT is unknown.
- 2D slices at 128×128 lose fine boundary detail and 3D context; a 3D model at native resolution would do better.
- The training patients have liver metastases; the model has not been checked on healthy anatomy or other pathologies.
- Uploaded images in the demo must already be axial CT in a soft-tissue window; other images give meaningless output.
