document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const processedPreview = document.getElementById('processedPreview');
    const resultContainer = document.getElementById('resultContainer');
    const predictionElement = document.getElementById('prediction');
    const loadingElement = document.getElementById('loading');
    const errorElement = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');

    // Sanity check for required elements
    const requiredElements = [
        {el: dropZone, id: 'dropZone'},
        {el: fileInput, id: 'fileInput'},
        {el: previewContainer, id: 'previewContainer'},
        {el: processedPreview, id: 'processedPreview'},
        {el: resultContainer, id: 'resultContainer'},
        {el: predictionElement, id: 'prediction'},
        {el: loadingElement, id: 'loading'},
        {el: errorElement, id: 'error'},
        {el: errorMessage, id: 'errorMessage'},
    ];

    for (const {el, id} of requiredElements) {
        if (!el) {
            console.error(`Element with id "${id}" not found!`);
            return; // Stop if any required element is missing
        }
    }

    // State variables
    let processing = false;
    let currentFile = null;

    // Initialize event listeners
    function init() {
        dropZone.addEventListener('click', handleZoneClick);

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlightDropZone);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlightDropZone);
        });

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        dropZone.addEventListener('drop', handleFileDrop);
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

        resetUI();
        processing = true;
        currentFile = file;
        loadingElement.style.display = 'block';

        try {
            if (!validateFile(file)) {
                processing = false;
                loadingElement.style.display = 'none';
                return;
            }

            await processWithTimeout(async () => {
                // Display processed preview (28x28 grayscale)
                await displayProcessedPreview(file);

                // Get prediction from server
                await getPrediction(file);
            }, 15000);

        } catch (error) {
            showError(error.message);
        } finally {
            processing = false;
            loadingElement.style.display = 'none';
        }
    }

    // Helper with timeout
    async function processWithTimeout(task, timeout) {
        let timeoutId;

        const timeoutPromise = new Promise((_, reject) => {
            timeoutId = setTimeout(() => {
                reject(new Error('Processing took too long. Try a smaller image.'));
            }, timeout);
        });

        try {
            await Promise.race([
                task(),
                timeoutPromise
            ]);
        } finally {
            clearTimeout(timeoutId);
        }
    }

    // File validation
    function validateFile(file) {
        if (!file) {
            showError('No file selected');
            return false;
        }

        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(file.type.toLowerCase())) {
            showError('Please upload a JPG, JPEG, or PNG image');
            return false;
        }

        const maxSize = 5 * 1024 * 1024; // 5MB
        if (file.size > maxSize) {
            showError('File size too large. Max 5MB allowed');
            return false;
        }

        return true;
    }

    // Display processed preview (28x28 grayscale)
    async function displayProcessedPreview(file) {
        return new Promise(async (resolve, reject) => {
            try {
                const mediumSizeUrl = await resizeImage(file, 200);
                const processedUrl = await resizeImage(mediumSizeUrl, 28, true);

                processedPreview.onload = () => {
                    previewContainer.style.display = 'flex';
                    resolve();
                };
                processedPreview.onerror = () => {
                    reject(new Error('Failed to load processed image'));
                };
                processedPreview.src = processedUrl;
            } catch (error) {
                reject(error);
            }
        });
    }

    // Generic image resizing function
    function resizeImage(source, size, grayscale = false) {
        return new Promise((resolve, reject) => {
            const img = new Image();

            img.onload = () => {
                try {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');

                    canvas.width = size;
                    canvas.height = size;

                    ctx.fillStyle = 'white';
                    ctx.fillRect(0, 0, size, size);

                    const scale = Math.min(
                        size / img.width,
                        size / img.height
                    );

                    const x = (size - img.width * scale) / 2;
                    const y = (size - img.height * scale) / 2;

                    ctx.drawImage(img, x, y, img.width * scale, img.height * scale);

                    if (grayscale) {
                        const imageData = ctx.getImageData(0, 0, size, size);
                        const data = imageData.data;

                        for (let i = 0; i < data.length; i += 4) {
                            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
                            data[i] = data[i + 1] = data[i + 2] = avg;
                        }

                        ctx.putImageData(imageData, 0, 0);
                    }

                    resolve(canvas.toDataURL());
                } catch (error) {
                    reject(error);
                }
            };

            img.onerror = () => {
                reject(new Error('Failed to load image for processing'));
            };

            if (typeof source === 'string') {
                img.src = source;
            } else {
                const reader = new FileReader();
                reader.onload = (e) => {
                    img.src = e.target.result;
                };
                reader.onerror = () => {
                    reject(new Error('Failed to read file for processing'));
                };
                reader.readAsDataURL(source);
            }
        });
    }

    // Get prediction from server
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

    // Display results
    function displayResults(data) {
        predictionElement.textContent = data.prediction;
        resultContainer.style.display = 'block';

        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
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
        previewContainer.style.display = 'none';
        processedPreview.src = '';
    }

    // Remove image handler
    window.removeImage = function() {
        if (processing) return;
        resetUI();
        currentFile = null;
    };

    // Initialize the application
    init();
});
