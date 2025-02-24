from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
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

# Load the trained model
model = MNISTNet()
model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
model.eval()

def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img_bytes = file.read()
    tensor = transform_image(img_bytes)
    outputs = model(tensor)
    _, predicted = torch.max(outputs, 1)
    return jsonify({'prediction': predicted.item()})

if __name__ == '__main__':
    app.run(debug=True)