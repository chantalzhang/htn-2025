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
    from sport_database import get_sport_info, get_sport_stats, get_sport_description, get_sport_name
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
                'weight': 0, 
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
        
        if not ML_AVAILABLE:
            # Provide mock response when ML modules aren't available
            print("ML modules not available, providing mock response")
            mock_recommendation = {
                'sport': 'Basketball',
                'stats': {
                    'strength': 75, 'agility': 85, 'endurance': 70, 'power': 80, 
                    'speed': 80, 'flexibility': 65, 'coordination': 85, 'balance': 80
                },
                'description': 'Helpful body traits: Height and wingspan give players a major advantage in rebounding, blocking, and shooting. Lean muscle mass and agility allow for explosive moves and quick changes of direction.'
            }
            
            mock_similar_athlete = {
                'name': 'Sample Athlete',
                'sport': 'Basketball',
                'measurements': {
                    'height': 185.0,
                    'weight': 80.0,
                    'wingspan': 190.0,
                    'shoulderWidth': 45.0,
                    'waist': 80.0,
                    'hip': 90.0
                }
            }
            
            return jsonify({
                'success': True,
                'data': {
                    'recommendation': mock_recommendation,
                    'similar_athlete': mock_similar_athlete
                }
            }), 200
        
        # Initialize recommendation engine
        engine = SportRecommendationEngine()
        
        # Get recommendation (using correct function signature)
        recommendation = engine.recommend_sport(
            gender=data['gender'],
            height_cm=data['height'],
            weight_kg=data['weight'],
            arm_span=data['wingspan'],
            leg_length=None,  # Optional parameter
            torso_length=None  # Optional parameter
        )
        
        # Get top sports list for filtering similar athletes
        all_sports = recommendation.get('all_top_sports', {})
        top_sports = []
        for sport_key, sport_count in all_sports.items():
            sport_info = get_sport_info(sport_key)
            top_sports.append(sport_info.get('name', sport_key))
        
        # Get similar athletes from top sports
        similar_athletes_result = engine.find_similar_athletes(
            gender=data['gender'],
            height_cm=data['height'],
            weight_kg=data['weight'],
            arm_span=data['wingspan'],
            leg_length=None,  # Optional parameter
            torso_length=None,  # Optional parameter
            top_sports=top_sports,  # Filter by top sports
            num_athletes=3  # Get 3 athletes
        )
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        recommendation_clean = convert_numpy_types(recommendation)
        similar_athletes_clean = convert_numpy_types(similar_athletes_result)
        
        # Format response according to the requested structure
        # Get all sports from the cluster
        all_sports = recommendation_clean.get('all_top_sports', {})
        
        # Create list of all sports in the cluster with their info
        cluster_sports = []
        for sport_key, sport_count in all_sports.items():
            sport_info = get_sport_info(sport_key)
            sport_stats = get_sport_stats(sport_key)
            sport_description = get_sport_description(sport_key)
            sport_name = get_sport_name(sport_key)
            
            cluster_sports.append({
                'sport': sport_name,
                'stats': sport_stats,
                'description': sport_description,
                'athlete_count': sport_count
            })
        
        # Sort by athlete count (most popular first)
        cluster_sports.sort(key=lambda x: x['athlete_count'], reverse=True)
        
        # The top sport should be the one actually recommended by the ML engine
        # Find the recommended sport in the cluster list
        recommended_sport_key = recommendation_clean['recommended_sport']
        top_sport = None
        
        for sport in cluster_sports:
            # Get the sport key for comparison
            sport_key = None
            for key, count in all_sports.items():
                sport_info = get_sport_info(key)
                if sport_info.get('name') == sport['sport']:
                    sport_key = key
                    break
            
            if sport_key == recommended_sport_key:
                top_sport = sport
                break
        
        # Fallback to first sport if recommended sport not found
        if not top_sport and cluster_sports:
            top_sport = cluster_sports[0]
        elif not top_sport:
            top_sport = {
                'sport': recommendation_clean['sport_name'],
                'stats': recommendation_clean['sport_stats'],
                'description': recommendation_clean['sport_description'],
                'athlete_count': 0
            }
        
        formatted_recommendation = {
            'top_sport': top_sport,
            'all_sports': cluster_sports
        }
        
        # Extract athlete data from the similar_athletes response
        similar_athletes_list = similar_athletes_clean.get('similar_athletes', [])
        formatted_similar_athletes = []
        
        for athlete_result in similar_athletes_list:
            athlete_data = athlete_result.get('athlete', {})
            
            # Get sport name from sport key
            sport_key = athlete_data.get('sport', '')
            sport_name = get_sport_name(sport_key) if sport_key else 'Unknown Sport'
            
            # Determine gender emoji
            gender = athlete_data.get('gender', '').lower()
            gender_emoji = '♂️' if gender == 'male' else '♀️' if gender == 'female' else '⚥'
            
            # Handle missing measurements
            def format_measurement(value):
                return value if value and value != 0 else "Information unavailable"
            
            formatted_athlete = {
                'name': athlete_data.get('Player', 'Unknown Athlete'),
                'sport': sport_name,
                'gender_emoji': gender_emoji,
                'measurements': {
                    'height': format_measurement(athlete_data.get('height_cm')),
                    'weight': format_measurement(athlete_data.get('weight_kg')),
                    'wingspan': format_measurement(athlete_data.get('Arm Span')),
                    'shoulderWidth': format_measurement(data.get('shoulderWidth')),
                    'waist': format_measurement(data.get('waist')),
                    'hip': format_measurement(data.get('hip'))
                }
            }
            formatted_similar_athletes.append(formatted_athlete)
        
        return jsonify({
            'success': True,
            'data': {
                'recommendation': formatted_recommendation,
                'similar_athletes': formatted_similar_athletes
            }
        }), 200
        
    except Exception as e:
        print(f"Error in recommend endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
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
