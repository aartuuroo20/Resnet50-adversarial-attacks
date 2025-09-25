# Adversarial Attacks on ResNet-50 (CelebA and VGGFace2)

This repository is a more **complete and extended version** of the repository Resnet50-adversarial-robustness. It contains two projects that analyze the **vulnerability of ResNet-50 models** trained on the **CelebA** and **VGGFace2** datasets, against adversarial attacks. Various attack techniques, robustness tests, and analysis on security in computer vision models are implemented.

## Repository Content

### **CelebA ResNet-50 Attacks**
- **Objective**: Evaluate adversarial attacks on a ResNet-50 model trained on the **CelebA** dataset.
- **Content**:
  - `resnet50_celeba_subset.pth`: Pre-trained model on CelebA.
  - `app.py`: Main application for attack testing.
  - `config.yaml`: Model and parameter configuration.
  - `access_control_app.py`: Code for controlling access to the model.
  - `scripts/`: Scripts for verifying, testing, and running attacks:
    - `verify.py`: Model verification.
    - `test_attacks.py`: Testing various adversarial attacks.
    - `attacks.py`: Implementation of several adversarial attacks.
    - `train.py`: Training the model on the CelebA dataset.
  - `Important_files/`: Essential files for the model:
    - `identity_CelebA.txt`: Information about CelebA identity.
    - `list_eval_partition.txt`: CelebA evaluation partition.

---

### **VGGFace2 ResNet-50 Attacks**
- **Objective**: Evaluate adversarial attacks on a ResNet-50 model trained on the **VGGFace2** dataset.
- **Content**:
  - `resnet50_vggface2hq_top10.pth`: Pre-trained model on VGGFace2.
  - `app.py`: Main application for attack testing.
  - `config.yaml`: Model and parameter configuration.
  - `access_control_app.py`: Code for controlling access to the model.
  - `scripts/`: Scripts for verifying, testing, and running attacks:
    - `verify.py`: Model verification.
    - `download_meta.py`: Download metadata.
    - `test_attacks.py`: Testing various adversarial attacks.
    - `download_vggface2hq.py`: Download the VGGFace2 HQ dataset.
    - `attacks.py`: Implementation of several adversarial attacks.
    - `train.py`: Training the model on the VGGFace2 dataset.

---

### Quick installation:
```bash
pip install -r requirements.txt
