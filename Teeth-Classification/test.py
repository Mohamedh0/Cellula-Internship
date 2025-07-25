import requests
import os
from collections import Counter

# API base URL 
BASE_URL = "http://localhost:8000"

# Directory with test images 
TEST_IMAGE_DIR = ""
CLASSES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

def test_health_endpoint():
    """Test the /health endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        print("Health Endpoint Test:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Health Endpoint Test Failed: {e}")

def test_frontend_endpoint():
    """Test the / endpoint (frontend)."""
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        print("Frontend Endpoint Test:")
        print(f"Status Code: {response.status_code}")
        print(f"Response contains 'Teeth Disease Classifier': {'Teeth Disease Classifier' in response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Frontend Endpoint Test Failed: {e}")

def test_predict_endpoint(image_path):
    """Test the /predict endpoint with an image."""
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return None, None
    
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": (os.path.basename(image_path), image_file, "image/jpeg")}
            response = requests.post(f"{BASE_URL}/predict", files=files)
            response.raise_for_status()
            result = response.json()
            print(f"Predict Endpoint Test for {image_path}:")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {result}")
            return result.get("predicted_class"), result.get("confidence")
    except requests.exceptions.RequestException as e:
        print(f"Predict Endpoint Test Failed for {image_path}: {e}")
        return None, None

def test_multiple_images(image_dir):
    """Test multiple images and show class distribution."""
    predictions = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                predicted_class, confidence = test_predict_endpoint(image_path)
                if predicted_class:
                    predictions.append(predicted_class)
    
    if predictions:
        print("\nPredicted class distribution:")
        print(Counter(predictions))
    else:
        print("No successful predictions")

if __name__ == "__main__":
    print("Starting API tests...\n")
    test_health_endpoint()
    print("\n" + "="*50 + "\n")
    test_frontend_endpoint()
    print("\n" + "="*50 + "\n")
    # test_multiple_images(TEST_IMAGE_DIR)
