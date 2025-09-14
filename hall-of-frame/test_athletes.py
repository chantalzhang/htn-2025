#!/usr/bin/env python3
import sys
sys.path.append('.')
from backend.app import app
import json

# Test the backend endpoint directly
with app.test_client() as client:
    test_data = {
        'gender': 'male',
        'height': 185.0,
        'weight': 80.0,
        'wingspan': 190.0,
        'shoulderWidth': 45.0,
        'waist': 80.0,
        'hip': 90.0
    }
    
    response = client.post('/recommend', json=test_data)
    print('Status Code:', response.status_code)
    if response.status_code == 200:
        data = response.get_json()
        print('Success:', data['success'])
        if data['success']:
            rec = data['data']['recommendation']
            athletes = data['data']['similar_athletes']
            print('Top Sport:', rec['top_sport']['sport'])
            print('All Sports:', [s['sport'] for s in rec['all_sports']])
            print('Number of Athletes:', len(athletes))
            for i, athlete in enumerate(athletes):
                print(f'Athlete {i+1}: {athlete["name"]} - {athlete["sport"]} {athlete["gender_emoji"]}')
                print(f'  Height: {athlete["measurements"]["height"]}')
                print(f'  Weight: {athlete["measurements"]["weight"]}')
        else:
            print('Error:', data.get('error'))
    else:
        print('Response:', response.get_json())
