# Plant Disease Detection

A four-class image-classification project using a pretrained ResNet18 and PlantVillage leaf images.

## Classes

- Apple scab (`Apple___Apple_scab`)
- Apple black rot (`Apple___Black_rot`)
- Tomato early blight (`Tomato___Early_blight`)
- Potato late blight (`Potato___Late_blight`)

## Dataset layout

Place the four class folders under `data/PlantVillage`:

```text
data/PlantVillage/
  Apple___Apple_scab/
  Apple___Black_rot/
  Tomato___Early_blight/
  Potato___Late_blight/
```

Each class folder should contain its labeled leaf images. The script makes stratified training, validation, and test splits automatically (80%, 10%, and 10%).

Download the four color-image classes from PlantVillage with:

```powershell
python download_plantvillage.py
```

The source dataset is available from [Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage) under CC BY-SA 3.0. Please cite Mohanty, Hughes, and Salathé (2016), “Using Deep Learning for Image-Based Plant Disease Detection,” *Frontiers in Plant Science*, DOI: 10.3389/fpls.2016.01419.

## Setup on Windows

Open this folder in VS Code, then run these commands in the integrated PowerShell terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, choose the `.venv` interpreter from VS Code's Python interpreter selector, or run `.venv\Scripts\python.exe` directly.

## Train and evaluate

```powershell
python plant_disease.py train --data-dir data/PlantVillage --epochs 10
```

The best checkpoint by validation accuracy is saved as `plant_model.pth`. The script prints test accuracy, a per-class precision/recall/F1 report, and the confusion matrix. The first training run downloads pretrained ResNet18 weights.

For a faster CPU demo, train the classifier head on a balanced subset:

```powershell
python plant_disease.py train --data-dir data/PlantVillage --epochs 3 --max-per-class 125 --freeze-backbone
```

This quick model uses fewer examples and a frozen feature extractor; its predictions are less reliable than a full training run.

## Compare traditional classifiers

To compare three traditional classifiers on the same stratified 80/10/10 sample split, run:

```powershell
python compare_classifiers.py --data-dir data/PlantVillage --max-per-class 125
```

The script extracts 512-dimensional features with a frozen ImageNet-pretrained ResNet18, then trains an RBF-kernel SVM, Random Forest, and Decision Tree on the training split. It prints test accuracy, macro-F1, and a confusion matrix for each classifier. The validation split is held out and the test split is not used during fitting. This comparison uses frozen ImageNet features, while the quick ResNet18 demo above trains its classification head on PlantVillage, so the results compare two different training approaches.

The split is stratified by image, not by leaf identity. PlantVillage contains multiple images of some leaves, so related images may land in both training and test splits; test scores may therefore be optimistic and should be treated as a classroom demo, not a field-accuracy estimate.

## Predict one image

```powershell
python plant_disease.py predict --image path\to\leaf.jpg --model plant_model.pth
```

The output includes the predicted class and model confidence. This is a learning/demo project; predictions on clean PlantVillage images may not generalize to field photos.

## Run in Streamlit

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

The app downloads the demo checkpoint from the `v1.0` GitHub release on first start. The model checkpoint and image dataset are intentionally kept out of Git commits. To deploy, create a Streamlit Community Cloud app from this repository's `main` branch and set the entry point to `app.py`. The model can be hosted elsewhere by setting `PLANT_MODEL_URL` to a direct checkpoint download URL.
