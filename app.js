document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const resultContainer = document.getElementById('outputSection');
    const predictionElement = document.getElementById('prediction');
    const loadingElement = document.getElementById('loading');
    const errorElement = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');

    // State
    let processing = false;
    let currentFile = null;

    function init() {
        dropZone.addEventListener('click', handleZoneClick);
        ['dragenter', 'dragover'].forEach(e => dropZone.addEventListener(e, highlightDropZone));
        ['dragleave', 'drop'].forEach(e => dropZone.addEventListener(e, unhighlightDropZone));
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => document.body.addEventListener(e, preventDefaults));
        dropZone.addEventListener('drop', handleFileDrop);
        fileInput.addEventListener('change', handleFileInput);
    }

    function handleZoneClick() {
        if (!processing) fileInput.click();
    }

    function handleFileDrop(e) {
        processFiles(e.dataTransfer.files);
    }

    function handleFileInput(e) {
        processFiles(e.target.files);
    }

    async function processFiles(files) {
        if (processing) return;
        const file = files[0];
        if (!file) return;

        resetUI();
        processing = true;
        currentFile = file;
        loadingElement.style.display = 'block';

        try {
            if (!validateFile(file)) return;

            await processWithTimeout(async () => {
                await displayOriginalImage(file);
                await getPrediction(file);
            }, 15000);

        } catch (error) {
            showError(error.message);
        } finally {
            processing = false;
            loadingElement.style.display = 'none';
        }
    }

    async function processWithTimeout(task, timeout) {
        let timeoutId;
        const timeoutPromise = new Promise((_, reject) => {
            timeoutId = setTimeout(() => reject(new Error('Processing took too long. Try a smaller image.')), timeout);
        });

        try {
            await Promise.race([task(), timeoutPromise]);
        } finally {
            clearTimeout(timeoutId);
        }
    }

    function validateFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        const maxSize = 5 * 1024 * 1024;
        if (!file) {
            showError('No file selected');
            return false;
        }
        if (!validTypes.includes(file.type.toLowerCase())) {
            showError('Please upload a JPG, JPEG, or PNG image');
            return false;
        }
        if (file.size > maxSize) {
            showError('File size too large. Max 5MB allowed');
            return false;
        }
        return true;
    }

    async function displayOriginalImage(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.onload = () => {
                    resultContainer.style.display = 'block';
                    resolve();
                };
                imagePreview.onerror = () => reject(new Error('Failed to load original image'));
                imagePreview.src = e.target.result;
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    }

    async function getPrediction(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            const response = await fetch('https://digit-recognizer-backend-production.up.railway.app/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server error');
            }

            const data = await response.json();
            if (data.success) {
                displayResults(data);
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        } catch (error) {
            throw new Error('Prediction error: ' + error.message);
        }
    }

    function displayResults(data) {
        predictionElement.textContent = data.prediction;
        resultContainer.style.display = 'block';
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function highlightDropZone() {
        dropZone.classList.add('dragover');
    }

    function unhighlightDropZone() {
        dropZone.classList.remove('dragover');
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorElement.style.display = 'block';
    }

    function resetUI() {
        errorElement.style.display = 'none';
        resultContainer.style.display = 'none';
        fileInput.value = '';
        imagePreview.src = '';
    }

    window.removeImage = function () {
        if (processing) return;
        resetUI();
        currentFile = null;
    };

    init();
});
