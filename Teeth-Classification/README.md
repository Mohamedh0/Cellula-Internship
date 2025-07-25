# Teeth Classification Web App

A Computer-Vision web application for classifying dental images into various medical conditions using a trained Keras model and FastAPI backend.

---

## 📸 App Preview

![App Screenshot](https://github.com/Mohamedh0/Teeth-Classification/blob/main/static/Screenshot%20(148).png)
---

## Features

- Upload dental images via a user-friendly web interface.
- Real-time image preview before prediction.
- Sends images to a FastAPI backend for classification.
- Displays predicted class with confidence score.
- Built with:
  - **FastAPI** (backend)
  - **TensorFlow/Keras** (model)
  - **HTML + JavaScript + CSS** (frontend)

---

## Project Structure

```
project/
├── main.py               
├── models/
│   └── best_model.h5     
├── notebooks/
│   ├── Teeth_Classification.ipynb       
│   └── Teeth-Classification-transferlearning.ipynb  
├── static/
│   ├── index.html        
│   ├── script.js         
│   ├── styles.css        
│   └── screenshot.png    
├── test.py               
├── README.md             
└── requirements.txt      
```

---

## ⚙️ How to Run the Project


### 1. Clone the Repository

```bash
git clone https://github.com/Mohamedh0/Teeth-Classification
cd Teeth-Classification
```

### 2. Create a virtual environment and install dependencies:

```bash
conda create -n teeth_classification
```

```bash
conda activate teeth_classification
```

```bash
pip install -r requirements.txt
```

### 3. Verify Project Setup

- Run the notebook of transferlearning and save the model to models folder 

### 4. Start the FastAPI Server

Run the FastAPI server to serve both the backend and frontend:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

```bash
cd static
```

```bash
python -m http.server 5500
```

The application will be available at:

- Frontend: `http://127.0.0.1:8000`
- API Health: `http://127.0.0.1:8000/health`
- API Predict: `http://127.0.0.1:8000/predict`

### 5. Test the Application

#### Test the Frontend

- Open `http://127.0.0.1:8000` in a browser.
- Upload a dental image (JPG/PNG) and click "Predict."
- Verify the image preview and prediction

#### Test the API

- Run the test script to verify API endpoints:

  ```bash
  python test.py
  ```
- Update `TEST_IMAGE_DIR` in `test.py` to point to your test images (e.g., `test_images/`).
- Expected output:

  ```
  Health Endpoint Test:
  Status Code: 200
  Response: {'status': 'Teeth Classification API is running'}
  
  Frontend Endpoint Test:
  Status Code: 200
  Response contains 'Teeth Disease Classifier': True
  
  Predict Endpoint Test for test_images/CaS/image1.jpg:
  Status Code: 200
  Response: {'predicted_class': 'CaS', 'confidence': 0.987654}
  ...
  Predicted class distribution:
  Counter({'CaS': 10, 'CoS': 8, 'Gum': 5, 'MC': 4, 'OC': 3, 'OLP': 2, 'OT': 1})
  ```

---

## 🧠 Model Details

- **Framework**: TensorFlow/Keras
- **Input Shape**: 224x224 pixels
- **Model Path**: `models/best_model.h5`
- **Classes**:
  - CaS – Candidiasis
  - CoS – Composite
  - Gum – Gum Disease
  - MC – Mucosal Condition
  - OC – Oral Cancer
  - OLP – Oral Lichen Planus
  - OT – Other

