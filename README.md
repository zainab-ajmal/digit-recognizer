# Image Classification Web Application

A simple web application that uses a pre-trained machine learning model to classify images. The application features a modern, responsive interface and supports drag-and-drop image uploads.

## Features

- Drag-and-drop image upload
- Real-time image preview
- Instant classification results
- Responsive design for mobile and desktop
- Error handling and loading states
- Support for PNG, JPG, and JPEG images

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure your model file (`my_model.keras`) is in the root directory of the project.

2. Start the Flask development server:
```bash
python app.py
```

3. Open your web browser and navigate to `http://localhost:5000`

## Deployment

### Deploying to Heroku

1. Create a `Procfile` in the root directory:
```
web: gunicorn app:app
```

2. Create a new Heroku app:
```bash
heroku create your-app-name
```

3. Deploy to Heroku:
```bash
git push heroku main
```

### Deploying to Vercel

1. Install the Vercel CLI:
```bash
npm install -g vercel
```

2. Create a `vercel.json` file in the root directory:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

3. Deploy to Vercel:
```bash
vercel
```

## Project Structure

```
.
├── app.py              # Flask application
├── my_model.keras      # Pre-trained model file
├── requirements.txt    # Python dependencies
├── static/            # Static files
│   ├── app.js         # Frontend JavaScript
│   ├── styles.css     # CSS styles
│   ├── index.html     # Main HTML file
│   └── upload-icon.svg # Upload icon
└── uploads/           # Temporary upload directory
```

## Notes

- The application uses a pre-trained Keras model for image classification
- The model expects input images to be resized to 224x224 pixels
- Maximum file upload size is set to 16MB
- The application supports PNG, JPG, and JPEG image formats

## License

MIT License 