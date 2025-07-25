from flask import Flask, request, render_template
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import tifffile
import segmentation_models_pytorch as smp

app = Flask(__name__)

# Get the main project directory 
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the main directory
UPLOAD_FOLDER = os.path.join(MAIN_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(MAIN_DIR, 'static', 'results')
MODEL_PATH = os.path.join(MAIN_DIR, 'models', 'unetplusplus_best.pth')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Verify model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure 'unetplusplus_best.pth' is in the 'models' folder within {MAIN_DIR}.")

# Load the pretrained UNet++ model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b3', 
        encoder_weights=None,     
        in_channels=12,          
        classes=1                
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Albumentations preprocessing pipeline for inference
preprocess = A.Compose([
    A.Resize(128, 128),  
    A.Normalize(
        mean=[0.5] * 12,  
        std=[0.5] * 12   
    ),
    ToTensorV2()  
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No image selected', 400
    
    # Save uploaded TIFF image
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)
    
    # Load 12-channel TIFF image
    try:
        image = tifffile.imread(input_path) 
        if image.shape[-1] != 12:
            return 'Uploaded image must have 12 channels', 400
        if image.shape[:2] != (128, 128):
            return 'Uploaded image must be 128x128 pixels', 400
    except Exception as e:
        return f'Error loading TIFF image: {str(e)}', 400
    
    # Preprocess image with Albumentations
    augmented = preprocess(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(device) 
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)  
        output = (output > 0.5).float() 
    
    # Post-process the output
    mask = output.squeeze().cpu().numpy()  
    mask = (mask * 255).astype(np.uint8)  
    mask_image = Image.fromarray(mask)
    
    # Upscale segmentation mask to 256x256 for visualization
    mask_image = mask_image.resize((256, 256), Image.NEAREST)  
    
    # Save the segmentation mask
    result_filename = f'result_{file.filename.split(".")[0]}.png'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    mask_image.save(result_path)
    
    vis_image = image[:, :, [2,1,0]]
    vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8) * 255
    vis_image = vis_image.astype(np.uint8)
    vis_image_pil = Image.fromarray(vis_image)
    
    # Upscale visualization to 256x256
    vis_image_pil = vis_image_pil.resize((256, 256), Image.BILINEAR)  
    vis_filename = f'vis_{file.filename.split(".")[0]}.png'
    vis_path = os.path.join(UPLOAD_FOLDER, vis_filename)
    vis_image_pil.save(vis_path)
    
    return render_template('index.html', 
                        original_image=vis_filename, 
                        segmented_image=result_filename)

if __name__ == '__main__':
    app.run(debug=True)