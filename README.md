# SPCBViTNet: Skin Cancer Classification with SPCB and Vision Transformers

This repository contains the code and resources for the SPCBViTNet model, a hybrid architecture combining Spatial Pyramid Convolutional Blocks (SPCB) with Vision Transformers for skin cancer classification. The model has been evaluated on two benchmark datasets — **HAM10000** and **PAD-UFES-20** — with preprocessing techniques such as hair removal applied to improve accuracy.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Datasets](#datasets)  
- [Preprocessing](#preprocessing)  
- [Data Splitting](#data-splitting)  
- [Model Training and Evaluation](#model-training-and-evaluation)  
- [Usage Instructions](#usage-instructions)  
- [Training Details](#training-details)  
- [Results](#results)  
- [FAQ](#faq)  
- [DOI Links](#doi-links)  

---

## Project Overview

SPCBViTNet integrates spatial pyramid convolutional blocks with transformer-based architectures to classify skin lesion images into diagnostic categories. The codebase includes preprocessing, data splitting, training, and ablation studies to evaluate different model variants and hyperparameters.

---

## Datasets

- **HAM10000** and **PAD-UFES-20** are public benchmark datasets for skin lesion classification.  
- Please download them from their official sources and place the raw data inside the `Dataset/` folder.  
- Metadata CSV files for each dataset are included in the `metadata/` folder.

---

## Preprocessing

- Run `preprocess.ipynb` to apply hair removal (using blackhat morphological operations) and image normalization on the raw datasets.  
- The preprocessed images will be saved back in the `Dataset/` folder (or a specified output folder).  

---

## Data Splitting

- The `splits.ipynb` notebook handles splitting the dataset into training and testing sets based on labels.  
- This creates CSV files with image IDs and labels to be used in model training.  

---

## Model Training and Evaluation

- Model notebooks are inside the `models/` directory, separated by dataset and architecture.  
- Includes training for SPCBViTNet variants with GoogLeNet, VGG16, DeiT, ResNet, DenseNet, and ViT.  
- Ablation studies on epochs and optimizers are under the `ablation_study/` subfolders.  
- To run, update the dataset path variable in each notebook to your local path.  

---

## Usage Instructions

1. Download datasets from official websites and place inside `Dataset/` folder.  
2. Run `preprocess.ipynb` for hair removal and normalization preprocessing.  
3. Run `splits.ipynb` to create train-test splits.  
4. Select your desired model notebook from `models/` and update dataset paths as needed.  
5. Execute the notebook to train and evaluate the model.  
6. Ablation notebooks allow experimenting with different hyperparameters.

---

## Training Details

- Training was done for 50 epochs with a batch size of 32 using the Adam optimizer.  
- Initial learning rate: 0.0001, with StepLR scheduler to decay the rate for stable convergence.  
- Regularization includes dropout (0.5) and weight decay to improve robustness.  
- Dataset split: 90% training, 10% testing.  
- Cross-Entropy loss used for multi-class classification.  
- Models trained on NVIDIA A100 GPU (Google Colab).  

---

## Results

- Comparative results for all models and ablation studies are saved in `Result/model_comparison.xlsx`.  
- ROC curves and other evaluation metrics demonstrate the robustness and clinical applicability of SPCBViTNet.  

---

## FAQ

**Q1: Where can I download the raw datasets?**  
A: Download HAM10000 and PAD-UFES-20 from their official websites (links below in DOI section). Raw datasets are not included here due to licensing.

**Q2: How do I run preprocessing?**  
A: Run `preprocess.ipynb` to perform hair removal and image normalization.

**Q3: Can I train models on my own data?**  
A: Yes, but ensure folder and metadata format matches. Update paths accordingly in notebooks.

**Q4: How do I reproduce the results?**  
A: Follow the order: download datasets → preprocessing → splitting → model training notebooks.

**Q5: What are the hardware requirements?**  
A: Trained on NVIDIA A100 GPUs, but smaller GPUs are also possible with reduced batch size/epochs.

---

## DOI Links

- **Code DOI:** https://doi.org/10.5281/zenodo.15533168  
- **HAM10000 Dataset:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000  
- **PAD-UFES-20 Dataset:** https://www.kaggle.com/datasets/mahdavi1202/skin-cancer

---

**Thank you for your interest in SPCBViTNet!**

---
