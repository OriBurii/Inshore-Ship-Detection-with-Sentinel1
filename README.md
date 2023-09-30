# Inshore-Ship-Detection-with-Sentinel1
Inshore ship detection is now big interest of harbor system monitoring or ship detection with satellite remotesensing systems. We used FBR Image(Frozen Background Reference Image) [1] and CFAR(Constant False Alarm Rate) algorithm to detect ships in inshore area with statistical method. And for deep learning we used CFAR-guided CNN [2] method with FBR image. In this tutorial, we combinded two methods mentiond before to make the better inshore ship detection result.

# OriBuri.py
OriBuri is SAR image processing python module for non professionals. It can read tiff files and h5 files of KOMPSAT-5(Korea Multi-Purpose Satellite 5) images.  It offers simple image pre-processing algorithms and som of spatial, temporal filtering algorithms. It also provides some of application algorithms such as CFAR algorithm, ratio based change detection algorithms.

# eeDownloader.ipynb
This code will help you download satellite images from google earth engine-api. You can download Sentinel data, preprocessing completed and you can save your memories. You can use this code with OpenSARLab (https://opensciencelab.asf.alaska.edu/portal/hub/login).

# Reference
[1] Detecting Ephemeral Objects in SAR Time-Series Using Frozen Background-Based Change Detection (2020, MDPI remote sensing)

[2] CFAR-guided Convolutional Neural Network for Large Scale Scene SAR Ship Detection (2023, IEEE Radar Conference)
