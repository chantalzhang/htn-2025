from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from werkzeug.utils import secure_filename
import uuid

# Add parent directory to path to import our ML modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sport_recommendation_engine import SportRecommendationEngine
    from sport_database import get_sport_info, get_sport_stats, get_sport_description
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML modules not available: {e}")
    ML_AVAILABLE = False
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def health_check():
    # print("yuh")
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Flask server is running'
    })

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Upload and process an image file.
    
    Expected form data:
    - 'image': The image file to upload
    
    Returns:
    - JSON response with processing results
    """
    print("=== UPLOAD ENDPOINT HIT ===")
    print(f"Request method: {request.method}")
    print(f"Request files: {list(request.files.keys())}")
    print(f"Request form: {list(request.form.keys())}")
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'success': False
            }), 400
        
        file = request.files['image']
        
        # Check if user selected a file
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
                'success': False
            }), 400
        
        if file:
            # Generate unique filename
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
            filename = secure_filename(unique_filename)
            
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # ============================================
            # YOUR IMAGE PROCESSING LOGIC GOES HERE
            # ============================================
            # 
            # Process the uploaded image here.
            # You have access to:
            # - file_path: Full path to the saved image file
            # - filename: The secure filename
            # - file: The original file object (if needed)
            #
            # For now, returning mock measurement data
            # Replace this with your actual computer vision/ML logic:
            
            # Mock measurement extraction (replace with actual CV logic)
            import random
    
            mock_measurements = {
                'height': round(random.uniform(160, 190), 1),
                'weight': round(random.uniform(60, 90), 1), 
                'wingspan': round(random.uniform(160, 200), 1),
                'shoulderWidth': round(random.uniform(35, 50), 1),
                'waist': round(random.uniform(70, 100), 1),
                'hip': round(random.uniform(80, 110), 1)
            }
            
            processing_results = {
                'message': 'Image processed successfully - measurements extracted',
                'filename': filename,
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'extracted_measurements': mock_measurements,
                'processing_method': 'computer_vision_analysis'
            }
            print("Processing results:", processing_results)
            # ============================================
            # END OF YOUR PROCESSING LOGIC SECTION
            # ============================================
            
            # Return success response
            return jsonify({
                'success': True,
                'data': processing_results
            }), 200
            
    except Exception as e:
        # Handle any unexpected errors
        return jsonify({
            'error': f'An error occurred while processing the image: {str(e)}',
            'success': False
        }), 500

@app.route('/recommend', methods=['POST'])
def recommend_sport():
    """
    Get sport recommendation and similar athlete based on body measurements.
    
    Expected JSON data:
    {
        "gender": "male" or "female",
        "height": 180.5,
        "weight": 75.2,
        "wingspan": 185.0,
        "shoulderWidth": 45.0,
        "waist": 80.0,
        "hip": 90.0
    }
    
    Returns:
    - JSON response with sport recommendation and similar athlete
    """
    print("=== RECOMMEND ENDPOINT HIT ===")
    
    if not ML_AVAILABLE:
        return jsonify({
            'error': 'ML modules not available',
            'success': False
        }), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['gender', 'height', 'weight', 'wingspan', 'shoulderWidth', 'waist', 'hip']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'success': False
                }), 400
        
        # Initialize recommendation engine
        engine = SportRecommendationEngine()
        
        # Get recommendation
        recommendation = engine.recommend_sport(
            gender=data['gender'],
            height=data['height'],
            weight=data['weight'],
            wingspan=data['wingspan'],
            shoulder_width=data['shoulderWidth'],
            waist=data['waist'],
            hip=data['hip']
        )
        
        # Get similar athlete
        similar_athlete = engine.find_similar_athlete(
            gender=data['gender'],
            height=data['height'],
            weight=data['weight'],
            wingspan=data['wingspan'],
            shoulder_width=data['shoulderWidth'],
            waist=data['waist'],
            hip=data['hip']
        )
        
        return jsonify({
            'success': True,
            'data': {
                'recommendation': recommendation,
                'similar_athlete': similar_athlete
            }
        }), 200
        
    except Exception as e:
        print(f"Error in recommend endpoint: {str(e)}")
        return jsonify({
            'error': f'An error occurred while getting recommendations: {str(e)}',
            'success': False
        }), 500

@app.route('/uploads/<filename>', methods=['GET'])
def get_uploaded_file(filename):
    """Serve uploaded files (optional - for testing purposes)."""
    print(f"=== GET UPLOADED FILE ENDPOINT HIT: {filename} ===")
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            from flask import send_file
            return send_file(file_path)
        else:
            return jsonify({
                'error': 'File not found',
                'success': False
            }), 404
    except Exception as e:
        return jsonify({
            'error': f'Error serving file: {str(e)}',
            'success': False
        }), 500

if __name__ == '__main__':
    print(f"Starting Flask server...")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"Max file size: {MAX_CONTENT_LENGTH / (1024*1024):.1f}MB")
    
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,       # Default Flask port
        debug=True       # Enable debug mode for development
    )
