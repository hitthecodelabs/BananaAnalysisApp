# BananaAnalysisApp

## Introduction
This repository contains a Flask-based web application designed for the analysis of bananas using deep learning and convolutional neural networks. The app utilizes a TensorFlow Lite model to perform image processing and classification to determine the quality of bananas for potential export or rejection.

## Features
- Image upload and processing using Flask.
- Banana detection and cropping algorithm.
- Resizing images maintaining aspect ratio.
- Deep learning model inference for banana classification.

## Installation
To run the application locally, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the app with `python app.py`.

## Usage
Once the application is running, navigate to `localhost:5000` in your web browser. Upload an image of a banana, and the application will classify it as either suitable for export or as a reject.

## Model
The deep learning model used in this application is a TensorFlow Lite model. It is not included in the repository and should be added with the name `model.tflite` in the root directory.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License
This project is open source and available under the [MIT License](LICENSE).
