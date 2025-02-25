# MNIST Digit Recognition

This is a simple web application that uses a trained model to predict the digit in an image. Users can drag-and-drop or upload images of handwritten digits, which are then processed by a neural network model trained on the MNIST dataset. The model predicts the digit in the image and returns the predicted result to the user.

## Features

- Drag-and-drop or upload images of handwritten digits
- Predicts the digit in the image using a neural network model
- Displays the pixel values of the image as seen by the model

## Requirements

- Python 3.6+
- Flask
- PyTorch
- NumPy
- Pillow
- Torchvision

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/mnist-digit-recognition.git
    cd mnist-digit-recognition
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```


## Usage

1. Start the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Drag-and-drop or upload an image of a handwritten digit to get the prediction.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.