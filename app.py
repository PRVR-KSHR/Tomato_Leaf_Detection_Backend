from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import timm
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["https://vit-agrinet.netlify.app", "http://localhost:5000", "http://localhost:3000"])

# Define the disease classes (alphabetical order - common in PyTorch datasets)
CLASSES = ['Early Blight', 'Healthy', 'Late Blight', 'Septoria']

# CBAM Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
    
    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.max(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)[0].unsqueeze(-1)
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        out = torch.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv(out))
        return x * out

class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ViTAgriNet Model
class ViTAgriNet(nn.Module):
    def __init__(self, num_classes=4, img_size=224, pretrained=False):
        super(ViTAgriNet, self).__init__()
        
        # Vision Transformer backbone
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        vit_features = self.vit.num_features  # 768 for vit_base
        
        # CBAM attention module
        self.cbam = CBAM(channels=vit_features, reduction=8)
        
        # Classification head - matches saved model structure
        self.head = nn.Sequential(
            nn.LayerNorm(vit_features),          # head.0
            nn.Linear(vit_features, 512),         # head.1
            nn.GELU(),                            # head.2
            nn.Dropout(0.3),                      # head.3
            nn.Linear(512, num_classes)           # head.4
        )
    
    def forward(self, x):
        # ViT feature extraction
        x = self.vit(x)  # [batch, 768]
        
        # Reshape for CBAM (needs 4D input)
        batch_size = x.size(0)
        x = x.unsqueeze(-1).unsqueeze(-1)  # [batch, 768, 1, 1]
        
        # Apply CBAM attention
        x = self.cbam(x)
        
        # Flatten back
        x = x.view(batch_size, -1)
        
        # Classification
        x = self.head(x)
        
        return x

# Load the model (lazy loading - only load on first prediction)
device = torch.device('cpu')  # Use CPU only to save memory
model = None
model_loaded = False

def load_model():
    global model, model_loaded, device
    if model_loaded:
        return model
    
    print("Loading ViTAgriNet model...")
    model = ViTAgriNet(num_classes=4, pretrained=False)
    
    model_path = os.path.join(os.path.dirname(__file__), 'ViTAgriNet_best.pth')
    
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1_000_000:
        hf_url = os.environ.get('HF_MODEL_URL')
        if hf_url and hf_url != 'https://huggingface.co/your-username/your-repo/resolve/main/ViTAgriNet_best.pth':
            print(f"Model file missing or too small ({os.path.getsize(model_path) if os.path.exists(model_path) else 0} bytes). Downloading from Hugging Face: {hf_url}")
            try:
                response = requests.get(hf_url, stream=True, timeout=600)
                response.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                print(f"Model downloaded successfully! Size: {os.path.getsize(model_path)} bytes")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise RuntimeError(f"Failed to download model: {e}")
        else:
            print("Model file not found and HF_MODEL_URL not set")
            raise RuntimeError("Model file not found and HF_MODEL_URL not set")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with weights_only=True...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print("Model loaded successfully with weights_only=True!")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return None
    
    model.to(device)
    model.eval()
    model_loaded = True
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return jsonify({
        'message': 'Tomato Leaf Disease Detection API',
        'status': 'running',
        'classes': CLASSES
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Load model on first prediction
        current_model = load_model()
        if current_model is None:
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Read and preprocess the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Transform the image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = current_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probabilities = probabilities[0].cpu().numpy()
            
            # Prepare response
            result = {
                'prediction': CLASSES[predicted.item()],
                'confidence': float(confidence.item() * 100),
                'all_probabilities': {
                    CLASSES[i]: float(all_probabilities[i] * 100)
                    for i in range(len(CLASSES))
                }
            }
            
            return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model_loaded})

@app.route('/ready', methods=['GET'])
def ready():
    return jsonify({'ready': True})

if __name__ == '__main__':
    print(f"Using device: {device}")
    print(f"Model classes: {CLASSES}")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
