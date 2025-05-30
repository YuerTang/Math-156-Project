# LightCBAM-ResNet  
*A Lightweight Attention-Enhanced Backbone for Camera Pose Estimation*

This repository presents our final project for UCLA’s MATH 156 course. We improve upon the original PoseNet framework by integrating the Convolutional Block Attention Module (CBAM) into a ResNet backbone for camera pose estimation. Our CBAM-augmented model demonstrates better convergence and generalization performance compared to baseline architectures like VGG16 and vanilla ResNet.
Full Report: [here](Math156_Final_Report.pdf)

---
---

## 🚀 Highlights

- Replaces PoseNet’s VGG16 backbone with ResNet-50 pluse CBAM attention modules 
- Compares learned vs. fixed loss weight (CameraPoseLoss vs. static MSE)  
- Shows faster convergence and less overfitting in CBAM-enhanced models  

---

## 🗂️ Project Structure
```
.
├── baselines/                     # Backbone model definitions
│   ├── ResNet50.py
│   ├── VGG16.py
│   └── __init__.py
│
├── config_folder/                # YAML config files for experiments
│   ├── cbam-ll.yaml              # CBAM + Learned Loss
│   ├── cbam-sl.yaml              # CBAM + Static Loss
│   ├── resnet50-ll.yaml          # ResNet50 + Learned Loss
│   ├── resnet50-sl.yaml          # ResNet50 + Static Loss
│   ├── vgg16-ll.yaml             # VGG16 + Learned Loss
│   └── vgg16-sl.yaml             # VGG16 + Static Loss
│
├── models/                       # Attention model and architecture
│   ├── __init__.py
│   └── resnet_attention.py
│
│   └── utils/                    # Custom loss functions
│       ├── __init__.py
│       └── losses.py
│
├── data_preparation.py           # Dataset class and preprocessing
├── pipeline.py                   # Training script (main entry point)
├── main.ipynb                    # Jupyter notebook for demo/testing
├── README.md                     # Project overview and documentation
├── .gitignore                    # Git ignore file
```

---

## ⚙️ Installation

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

We use the **King’s College dataset** from the original PoseNet paper.

- Make sure the dataset is downloaded from [here](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3).  
- Set the correct path in your config file under `configs/*.yaml`.

---

## ▶️ How to Run

1. Prepare the dataset directory.  
2. Choose and adjust a config file (e.g., `configs/resnet_cbam.yaml`).  
3. Run the training:

```bash
python pipeline.py --config configs/resnet_cbam.yaml
```

4. Training logs and loss figures will be saved in the `figures/` folder.

---

## 📈 Results

Loss curves show:

- **CBAM accelerates convergence** — loss drops faster in early epochs.  
- **Better generalization** — ResNet+CBAM keeps training and validation loss close.  
- **Learned CameraPoseLoss** outperforms static loss in both smoothness and final values.
---


## 📚 References
The inituition and implementation of ResNet with CBAM are from the papers [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) and [ASA for Multi-Scene APR: Activating Self-Attention for Multi-Scene Absolute Pose Regression](https://arxiv.org/abs/2411.01443).
The camera pose estimation topic was inspired from [PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization][https://arxiv.org/abs/1505.07427](https://arxiv.org/abs/1505.07427)

---



