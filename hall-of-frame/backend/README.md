# Backend Flask Server

This Flask server provides an image upload API endpoint for processing images.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /upload
Upload an image file for processing.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Form data with 'image' field containing the image file

**Supported file types:** PNG, JPG, JPEG, GIF, BMP, WEBP
**Max file size:** 16MB

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Processing results here",
    "filename": "unique_filename.jpg",
    "file_path": "uploads/unique_filename.jpg"
  }
}
```

### GET /
Health check endpoint.

### GET /uploads/<filename>
Serve uploaded files (for testing purposes).

## Image Processing

Add your image processing logic in the designated section in `app.py` between the comments:
```python
# ============================================
# YOUR IMAGE PROCESSING LOGIC GOES HERE
# ============================================
```

The uploaded image file path is available as `file_path` variable.
