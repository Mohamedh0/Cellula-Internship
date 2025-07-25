const apiBaseUrl = window.location.origin;

async function predict() {
    const input = document.getElementById('imageInput');
    const preview = document.getElementById('preview');
    const resultDiv = document.getElementById('result');
    
    if (!input.files[0]) {
        resultDiv.innerHTML = "Please select an image.";
        return;
    }

    // Display image preview
    const img = document.createElement('img');
    img.src = URL.createObjectURL(input.files[0]);
    img.style.display = 'block';
    preview.innerHTML = '';
    preview.appendChild(img);

    const formData = new FormData();
    formData.append("file", input.files[0]);

    try {
        resultDiv.innerHTML = "Predicting...";
        const response = await fetch(`${apiBaseUrl}/predict`, {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        if (response.ok) {
            resultDiv.innerHTML = `Predicted Class: ${result.predicted_class} (Confidence: ${(result.confidence * 100).toFixed(2)}%)`;
        } else {
            resultDiv.innerHTML = `Error: ${result.detail}`;
        }
    } catch (error) {
        resultDiv.innerHTML = `Error: ${error.message}`;
    }
}
