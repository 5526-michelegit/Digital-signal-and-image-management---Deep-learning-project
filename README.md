
# Audio Classification & Face Generation with Deep Learning

## Description
This project tackles two complementary deep‑learning challenges:  
1. **Audio Classification**: preprocessing and feature extraction on the CMU Arctic speech dataset to train a neural model that classifies audio samples with over 99% accuracy.  
2. **Image Generation with GANs**: face detection, cropping, and training of Generative Adversarial Networks on the Labeled Faces in the Wild dataset to synthesize realistic human faces.

## Table of Contents
1. [Repository Structure](#repository-structure)  
2. [Requirements](#requirements)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Datasets](#datasets)  
6. [Objectives](#objectives)  
7. [Methodology](#methodology)  
8. [Tools & Libraries](#tools--libraries)  
9. [Achievements](#achievements)  
10. [Future Directions](#future-directions)  

## Repository Structure
```text
├── Progetto_DSIM_AUDIO.ipynb      # Notebook: audio classification pipeline (preprocessing → model training → evaluation)  
├── progetto_DSIM_IMAGE.ipynb      # Notebook: image preprocessing, face detection, GAN training & sample generation  
├── MODELLO_H5                     # Trained Keras model for audio classification (.h5 format)  
├── DSIM_PROJECT_PRESENTATION.pdf  # Slide deck summarizing objectives, methods, and results  
└── README.md                      # This documentation file  
```

## Requirements

* **Python 3.7+**
* **Jupyter Notebook**
* **Key Python packages** (install via `pip`):

  * Audio: `librosa`, `scipy`, `numpy`, `pandas`, `tensorflow`, `keras`
  * Image: `opencv‑python`, `dlib` (or `face_recognition`), `pillow`, `tensorflow`, `keras`
  * Utilities: `matplotlib`, `seaborn`, `scikit‑learn`

## Installation

1. **Clone** the repository:

   ```bash
   git clone https://github.com/5526-michelegit/Digital-signal-and-image-management---Deep-learning-project.git
   cd Digital-signal-and-image-management---Deep-learning-project
   ```
2. **(Optional)** Create & activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux/Mac
   venv\Scripts\activate.bat      # Windows
   ```
3. **Install** the required packages:

   ```bash
   pip install librosa scipy numpy pandas tensorflow keras opencv-python dlib pillow matplotlib seaborn scikit-learn
   ```

## Usage

1. **Launch** Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
2. **Audio Classification**

   * Open `Progetto_DSIM_AUDIO.ipynb`
   * Run cells to load CMU Arctic data, extract features (MFCCs, spectral), train the Keras model, and evaluate accuracy.
3. **Image Generation with GANs**

   * Open `progetto_DSIM_IMAGE.ipynb`
   * Run cells to detect & crop faces, preprocess images, define GAN architecture (e.g., DCGAN), train, and visualize generated samples.

## Datasets

* **CMU Arctic**: English speech corpus used for audio classification (download separately and point the notebook to your local path).
* **Labeled Faces in the Wild (LFW)**: Collection of face images for GAN training (download via `sklearn.datasets.fetch_lfw_people` or manually place in a folder).

## Objectives

* Develop a robust audio classification pipeline achieving >99% accuracy on CMU Arctic samples.
* Build and train GAN models capable of generating realistic human face images.
* Compare different network architectures and training strategies for both tasks.
* Package trained audio model (`MODELLO_H5`) for easy inference.

## Methodology

1. **Audio Pipeline**

   * Load WAV files → normalize → extract MFCCs and spectral features.
   * Define and compile a deep neural network in Keras (CNN or LSTM).
   * Train with train/validation split, monitor accuracy & loss curves.
2. **Image Pipeline**

   * Detect faces using OpenCV (Haar cascades) or dlib → crop & resize to 64×64.
   * Design a DCGAN: generator and discriminator architectures in Keras.
   * Train adversarially, save model checkpoints, and periodically sample outputs.
3. **Evaluation & Visualization**

   * Plot confusion matrix, ROC curves for audio classifier.
   * Display GAN-generated face grids and track loss metrics over epochs.

## Tools & Libraries

* **TensorFlow & Keras**: model definition and training
* **Librosa & SciPy**: audio loading & feature extraction
* **OpenCV & dlib**: face detection and image preprocessing
* **Matplotlib & Seaborn**: result visualization
* **Scikit‑Learn**: dataset utilities, metrics, and preprocessing helpers

## Achievements

* Achieved **99.12%** accuracy in audio classification, with precision and recall both above 99%.
* Successfully trained GANs that generate **realistic** face images indistinguishable at first glance.
* Packaged and saved the best audio model (`MODELLO_H5`) for downstream inference.

## Future Directions

* **Hyperparameter Optimization**: automate tuning with Keras Tuner or Optuna.
* **Data Augmentation**: apply audio perturbations and image transforms to improve generalization.
* **Advanced GANs**: experiment with StyleGAN or conditional GANs for higher-fidelity images.
* **Deployment**: wrap audio classifier into a REST API; deploy GAN in a demo web interface.
