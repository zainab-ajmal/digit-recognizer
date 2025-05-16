import os
import json
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
from torchvision import transforms
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Define the model architecture
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

# Initialize model
device = torch.device('cpu')  # Vercel only supports CPU
model = MNISTNet().to(device)

# Load model weights
model_path = os.path.join(os.path.dirname(__file__), '..', 'my_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            # Parse the multipart form data
            boundary = self.headers['Content-Type'].split('=')[1].encode()
            parts = post_data.split(boundary)
            
            # Find the file part
            file_data = None
            for part in parts:
                if b'filename=' in part:
                    file_data = part.split(b'\r\n\r\n')[1].split(b'\r\n')[0]
                    break
            
            if not file_data:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No file provided'}).encode())
                return
            
            # Process the image
            processed_image = preprocess_image(file_data)
            processed_image = processed_image.to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(processed_image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get the predicted class and confidence
            predicted_class = predicted.item()
            confidence = confidence.item()
            
            # Prepare response
            response = {
                'prediction': str(predicted_class),
                'confidence': confidence,
                'success': True
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode()) 