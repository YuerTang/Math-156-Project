# LightCBAM-ResNet  
*A Lightweight Attention-Enhanced Backbone for Camera Pose Estimation*

This repository presents our final project for UCLA’s MATH 156 course. We improve upon the original PoseNet framework by integrating the Convolutional Block Attention Module (CBAM) into a ResNet backbone for camera pose estimation. Our CBAM-augmented model demonstrates better convergence and generalization performance compared to baseline architectures like VGG16 and vanilla ResNet.

---

## 🚀 Highlights

- Replaces PoseNet’s VGG16 backbone with ResNet-50 pluse CBAM attention modules 
- Compares learned vs. fixed loss weight (CameraPoseLoss vs. static MSE)  
- Shows faster convergence and less overfitting in CBAM-enhanced models  

---

## 🗂️ Project Structure

```
├── configs/                # YAML config files for each experiment
├── data_preparation.py    # Dataset loading and preprocessing
├── pipeline.py            # Main training script
├── main.ipynb             # Sample experiment notebook
├── figures/               # Loss curve visualizations
├── utils/                 # Custom loss functions and CBAM modules
└── README.md              # This documentation
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

### 📉 Example Visualizations

| Learned Loss Log | Static Loss Log |
|------------------|-----------------|
| ![learned](figures/Learned_Loss_Logarithmic.png) | ![static](figures/Static_Loss_Logarithmic.png) |

---

## 🧱 Architecture Diagrams

### ResNet-18 + CBAM

> Could consider adding one CBAM attention module before each residual block.

![ResNet18](https://github.com/user-attachments/assets/1a402320-1396-427d-9ce8-76ddc91f4e00)

---

### Baseline: VGG-16 (PoseNet)

![VGG16](https://github.com/user-attachments/assets/b72c282c-1ba5-48e8-842b-cd5eef630ead)

---

## 📚 References

- Kendall et al., *PoseNet*  
  [https://arxiv.org/abs/1505.07427](https://arxiv.org/abs/1505.07427)

- Woo et al., *CBAM: Convolutional Block Attention Module*  
  [https://arxiv.org/abs/1807.06521](https://arxiv.org/abs/1807.06521)

---

## 📎 Report

📄 You can find the full technical write-up [here](file:///Users/yuertang/Downloads/Math_156%20(2).pdf)

---

