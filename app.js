document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const resultContainer = document.getElementById('resultContainer');
    const predictionElement = document.getElementById('prediction');
    const loadingElement = document.getElementById('loading');
    const errorElement = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');

    // State variables
    let processing = false;
    let currentFile = null;

    // Initialize event listeners
    function init() {
        // Click handler for upload area
        dropZone.addEventListener('click', handleZoneClick);

        // Drag and drop events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlightDropZone);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlightDropZone);
        });

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        // File drop handler
        dropZone.addEventListener('drop', handleFileDrop);

        // File input handler
        fileInput.addEventListener('change', handleFileInput);
    }

    // Event handlers
    function handleZoneClick() {
        if (!processing) {
            fileInput.click();
        }
    }

    function handleFileDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        processFiles(files);
    }

    function handleFileInput(e) {
        processFiles(e.target.files);
    }

    // Main processing function
    async function processFiles(files) {
        if (processing) return;
        
        const file = files[0];
        if (!file) return;

        // Reset UI and set processing state
        resetUI();
        processing = true;
        currentFile = file;
        loadingElement.style.display = 'block';

        try {
            // Validate file first
            if (!validateFile(file)) {
                return;
            }

            // Display original image
            await displayOriginalImage(file);
                
            // Get prediction from server
            await getPrediction(file);

        } catch (error) {
            showError(error.message);
        } finally {
            processing = false;
            loadingElement.style.display = 'none';
        }
    }

    // File validation
    function validateFile(file) {
        // Check if file exists
        if (!file) {
            showError('No file selected');
            return false;
        }

        // Check file type
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(file.type.toLowerCase())) {
            showError('Please upload a JPG, JPEG, or PNG image');
            return false;
        }

        // Check file size (max 5MB)
        const maxSize = 5 * 1024 * 1024; // 5MB
        if (file.size > maxSize) {
            showError('File size too large. Max 5MB allowed');
            return false;
        }

        return true;
    }

    // Display original image
    async function displayOriginalImage(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                imagePreview.onload = () => {
                    previewContainer.style.display = 'block';
                    resolve();
                };
                imagePreview.onerror = () => {
                    reject(new Error('Failed to load image'));
                };
                imagePreview.src = e.target.result;
            };
            
            reader.onerror = () => {
                reject(new Error('Failed to read file'));
            };
            
            reader.readAsDataURL(file);
        });
    }

    // Get prediction from server
    async function getPrediction(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/predict', {
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

    // Display results
    function displayResults(data) {
        predictionElement.textContent = data.prediction;
        resultContainer.style.display = 'block';
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // Drag and drop visual feedback
    function highlightDropZone() {
        dropZone.classList.add('dragover');
    }

    function unhighlightDropZone() {
        dropZone.classList.remove('dragover');
    }

    // Prevent default behaviors
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorElement.style.display = 'block';
    }

    // Reset UI
    function resetUI() {
        errorElement.style.display = 'none';
        resultContainer.style.display = 'none';
        fileInput.value = '';
    }

    // Remove image handler
    window.removeImage = function() {
        if (processing) return;
        resetUI();
        previewContainer.style.display = 'none';
        imagePreview.src = '';
        currentFile = null;
    };

    // Initialize the application
    init();
});
