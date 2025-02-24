from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import logging
from torchvision import transforms

app = Flask(__name__)

class MNISTNet(nn.Module):
    def __init__(self, hidden_size=128):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten images to [batch_size, 784]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = MNISTNet()
model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
model.eval()


def get_images():
    image_dir = 'images'
    image_files = []
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                image_files.append(filename)
    return image_files

@app.route('/')
def index():
    # Pass images to the template
    images = get_images()
    return render_template('index.html', images=images)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file)
        img = img.resize((28, 28))
        img = img.convert('L')
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1,1,28,28)

        img_tensor = torch.from_numpy(img_array).float()

        with torch.no_grad():
            output = model(img_tensor)
            predicted_digit = torch.argmax(output, dim=1).item()

        return jsonify({'prediction': str(predicted_digit), 'array': img_array.tolist()})
    except Exception as e:
        logging.error(f'Error during prediction: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)