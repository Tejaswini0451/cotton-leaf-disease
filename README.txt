# Cotton Plant Disease Detection Using Deep Learning

## Project Description
This project detects and classifies cotton leaf diseases using deep learning techniques. It predicts the disease category from an uploaded cotton leaf image through a Streamlit web application.

## Algorithms Used
1. Custom Convolutional Neural Network (CNN)
2. ResNet18 Transfer Learning
3. Vision Transformer (ViT)
4. Grad-CAM for model explainability

## Disease Classes
- Bacterial Blight
- Curl Virus
- Fusarium Wilt
- Healthy

## Technologies Used
- Python
- PyTorch
- Streamlit
- OpenCV
- PIL
- NumPy

## Project Structure
```text
cotton-leaf-disease/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── classes.json
    ├── cotton_cnn.pth
    ├── cotton_resnet18.pth
    └── cotton_vit.pth
```

## Installation and Execution

1. Clone the repository:
```bash
git clone https://github.com/Tejaswini0451/cotton-leaf-disease.git
```

2. Navigate to the project directory:
```bash
cd cotton-leaf-disease
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:
```bash
streamlit run app.py
```

5. Upload a cotton leaf image to obtain the disease prediction.

## Model Files
The trained CNN, ResNet18 and Vision Transformer models are stored in the `models` directory.

## Project Outcome
The application provides a comparative implementation of three deep learning models for cotton leaf disease classification.
