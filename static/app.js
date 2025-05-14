document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const processedPreview = document.getElementById('processedPreview');
    const resultContainer = document.getElementById('resultContainer');
    const predictionElement = document.getElementById('prediction');
    const confidenceElement = document.getElementById('confidence');
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

            // Process in chunks with timeouts to prevent UI freeze
            await processWithTimeout(async () => {
                // Display original image
                await displayOriginalImage(file);
                
                // Process and display preview
                await displayProcessedPreview(file);
                
                // Get prediction from server
                await getPrediction(file);
            }, 15000); // 15 second timeout

        } catch (error) {
            showError(error.message);
        } finally {
            processing = false;
            loadingElement.style.display = 'none';
        }
    }

    // Helper function with timeout
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
                    resolve();
                };
                imagePreview.onerror = () => {
                    reject(new Error('Failed to load original image'));
                };
                imagePreview.src = e.target.result;
            };
            
            reader.onerror = () => {
                reject(new Error('Failed to read file'));
            };
            
            reader.readAsDataURL(file);
        });
    }

    // Display processed preview (28x28 grayscale)
    async function displayProcessedPreview(file) {
        return new Promise(async (resolve, reject) => {
            try {
                // First create a smaller version (200px) to make final processing faster
                const mediumSizeUrl = await resizeImage(file, 200);
                
                // Then create the final 28x28 version
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
                    
                    // Set canvas dimensions
                    canvas.width = size;
                    canvas.height = size;
                    
                    // Fill with white background
                    ctx.fillStyle = 'white';
                    ctx.fillRect(0, 0, size, size);
                    
                    // Calculate scaling while maintaining aspect ratio
                    const scale = Math.min(
                        size / img.width,
                        size / img.height
                    );
                    
                    const x = (size - img.width * scale) / 2;
                    const y = (size - img.height * scale) / 2;
                    
                    // Draw the image
                    ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
                    
                    // Convert to grayscale if requested
                    if (grayscale) {
                        const imageData = ctx.getImageData(0, 0, size, size);
                        const data = imageData.data;
                        
                        for (let i = 0; i < data.length; i += 4) {
                            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
                            data[i] = data[i + 1] = data[i + 2] = avg;
                        }
                        
                        ctx.putImageData(imageData, 0, 0);
                    }
                    
                    // Return as data URL
                    resolve(canvas.toDataURL());
                } catch (error) {
                    reject(error);
                }
            };
            
            img.onerror = () => {
                reject(new Error('Failed to load image for processing'));
            };
            
            // Handle both file objects and data URLs
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
            
            const response = await fetch('/predict', {
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
        confidenceElement.textContent = `${(data.confidence * 100).toFixed(2)}%`;
        resultContainer.style.display = 'block';
        
        // Smooth scroll to results
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
    }

    // Remove image handler
    window.removeImage = function() {
        if (processing) return;
        resetUI();
        previewContainer.style.display = 'none';
        imagePreview.src = '';
        processedPreview.src = '';
        currentFile = null;
    };

    // Initialize the application
    init();
});