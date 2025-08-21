🧬 Malaria Cell Image Classifier

project Link for Interactive graphs inside [Link Text](file:///C:/Users/Aman%20Yadav/Downloads/Image_classifier.html)

A deep learning project using Convolutional Neural Networks (CNNs) to classify red blood cell (RBC) images as infected (parasitized) or uninfected.
Trained on 2000+ microscopic blood smear images from a publicly available Kaggle dataset, this model demonstrates the application of AI in biomedical image analysis and malaria detection.

🚀 Features

Classifies RBCs into Infected or Uninfected.

Built with TensorFlow/Keras.

Trained on 2000+ images for reliable performance.

Includes data preprocessing, training, and evaluation pipeline.

Demonstrates the use of CNNs in medical image classification.

📂 Dataset

Source: Kaggle - Malaria Cell Images Dataset

Contains labeled microscopic images of parasitized and uninfected cells.

🏗️ Model Architecture

Convolutional Layers (Conv2D) for feature extraction

MaxPooling Layers to reduce spatial dimensions

Flatten & Dense Layers for classification

Dropout for regularization

Binary classification output (Infected vs. Uninfected)

⚙️ Installation & Usage

Clone the repository:

git clone https://github.com/your-username/malaria-rbc-classifier.git
cd malaria-rbc-classifier


Install dependencies:

pip install -r requirements.txt


Run the notebook or script to train/predict:

jupyter notebook Image_classifier.ipynb

📊 Results

The trained CNN successfully classifies malaria-infected RBCs with high accuracy.

Sample prediction output:

Image	Predicted Label
🩸	Infected
🩸	Uninfected
🔮 Future Improvements

Add more training data for better generalization.

Implement advanced architectures (ResNet, EfficientNet).

Deploy as a web app for real-time malaria detection.

🙌 Acknowledgments

Dataset: Kaggle Malaria Cell Images

Tools: TensorFlow, Keras, OpenCV, Matplotlib

👤 Author

Aman Yadav
📌 Master’s Student in Life Science Informatics

