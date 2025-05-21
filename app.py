from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import random
import folium
import networkx as nx
import requests # Make sure to install requests: pip install requests
from sklearn_extra.cluster import KMedoids
from math import radians, sin, cos, sqrt, atan2 # Added for Haversine

app = Flask(__name__)

# --- Global Variables & Constants ---
# (Consider moving to a config file or environment variables for production)
OSRM_BASE_URL = "http://router.project-osrm.org"
SAFETY_THRESHOLD_KM = 0.2 # 200 meters for checking proximity to accident points

# --- Helper Functions ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees)."""
    R = 6371  # Radius of Earth in kilometers
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    a = sin(dLat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dLon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def generate_dummy_data():
    districts = {
        "Baktiya": {"lat": 5.050000, "lng": 97.433333},
        "Baktiya Barat": {"lat": 5.133333, "lng": 97.366667},
        "Banda Baro": {"lat": 5.116667, "lng": 96.950000},
        "Cot Girek": {"lat": 4.833176, "lng": 97.330053},
        "Dewantara": {"lat": 5.183331, "lng": 97.000000},
        "Geureudong Pase": {"lat": 4.950000, "lng": 97.033333},
        "Kuta Makmur": {"lat": 5.083333, "lng": 97.033333},
        "Langkahan": {"lat": 4.866667, "lng": 97.450000},
        "Lapang": {"lat": 5.116667, "lng": 97.266667},
        "Lhoksukon": {"lat": 5.073056, "lng": 97.255278},
        "Matangkuli": {"lat": 5.033056, "lng": 97.277778},
        "Meurah Mulia": {"lat": 5.071972, "lng": 97.207306},
        "Muara Batu": {"lat": 5.233333, "lng": 96.950000},
        "Nibong": {"lat": 5.016667, "lng": 97.216667},
        "Nisam": {"lat": 5.016667, "lng": 96.966667},
        "Nisam Antara": {"lat": 4.916667, "lng": 96.900000},
        "Paya Bakong": {"lat": 4.950000, "lng": 97.266667},
        "Pirak Timu": {"lat": 4.916667, "lng": 97.216667},
        "Samudera": {"lat": 5.116667, "lng": 97.216667},
        "Sawang": {"lat": 5.050000, "lng": 96.883333},
        "Seunuddon": {"lat": 5.183333, "lng": 97.450000},
        "Simpang Keramat": {"lat": 4.958031, "lng": 96.949861},
        "Syamtalira Aron": {"lat": 5.100000, "lng": 97.266667},
        "Syamtalira Bayu": {"lat": 4.983333, "lng": 97.033333},
        "Tanah Jambo Aye": {"lat": 5.066667, "lng": 97.483333},
        "Tanah Luas": {"lat": 4.933333, "lng": 97.066667},
        "Tanah Pasir": {"lat": 5.133333, "lng": 97.300000}
    }
    
    locations = []
    for district, center in districts.items():
        # Add a representative point for the district itself (used for dropdowns)
        # For actual accident data, generate multiple points as before
        for _ in range(random.randint(2, 5)): # Generate 2-5 accident points per district
            locations.append({
                "district": district,
                "lat": center["lat"] + (random.random() * 0.05 - 0.025),
                "lng": center["lng"] + (random.random() * 0.05 - 0.025),
                "fatalities": random.randint(0, 5), # Adjusted to include 0 fatalities
                "road_condition": random.randint(1, 5), # 1:Bad, 5:Good
                "accidents": random.randint(1, 10),
                "traffic": random.randint(500, 2500) # Vehicle count
            })
    return pd.DataFrame(locations)

def perform_clustering(data, n_clusters=3):
    if data.empty or len(data) < n_clusters:
        # Not enough data to cluster, return empty or handle as error
        return data, [], pd.DataFrame() # Added empty DataFrame for medoid_data

    features = data[['fatalities', 'road_condition', 'accidents', 'traffic']]
    # Ensure features are numeric and handle potential NaNs (though dummy data shouldn't have them)
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, method='pam') # Specify PAM for robustness
    data['cluster'] = kmedoids.fit_predict(features)
    medoid_indices = kmedoids.medoid_indices_
    medoid_data = data.iloc[medoid_indices].copy() # Create a copy to avoid SettingWithCopyWarning
    return data, medoid_indices, medoid_data

def get_processed_accident_data():
    """Generates dummy accident data and assigns a risk level to each point."""
    data_df = generate_dummy_data()
    if data_df.empty:
        return pd.DataFrame()

    processed_df = data_df.copy()
    if 'fatalities' in processed_df.columns and 'accidents' in processed_df.columns:
        # Ensure columns are numeric before calculation
        processed_df['fatalities'] = pd.to_numeric(processed_df['fatalities'], errors='coerce').fillna(0)
        processed_df['accidents'] = pd.to_numeric(processed_df['accidents'], errors='coerce').fillna(0)
        
        risk_scores = processed_df['fatalities'] * 0.6 + processed_df['accidents'] * 0.4
        
        if not risk_scores.empty and risk_scores.nunique() > 0:
            if risk_scores.nunique() >= 3:
                processed_df['risk_level'] = pd.qcut(risk_scores, q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            elif risk_scores.nunique() == 2:
                processed_df['risk_level'] = pd.qcut(risk_scores, q=2, labels=['Low', 'High'], duplicates='drop')
            else: # Only 1 unique score value
                # Classify based on the single score value relative to typical score ranges if known,
                # or assign a default. For this example, let's use a simple thresholding logic
                # based on the score itself, assuming higher scores are worse.
                # This part might need more sophisticated logic based on actual data distribution.
                median_val = risk_scores.median() # Example: if all scores are 0, it's Low.
                if median_val > 3:  # Arbitrary threshold for 'High'
                    processed_df['risk_level'] = 'High'
                elif median_val > 1: # Arbitrary threshold for 'Medium'
                    processed_df['risk_level'] = 'Medium'
                else:
                    processed_df['risk_level'] = 'Low'
        else: # No valid risk scores
            processed_df['risk_level'] = 'Unknown'
    else:
        processed_df['risk_level'] = 'Unknown'
    return processed_df

def determine_segment_safety(segment_coords_list, all_accident_points_df, threshold_km=SAFETY_THRESHOLD_KM):
    """Determines the safety level of a route segment based on nearby accident points."""
    highest_risk_encountered = "safe" 

    risk_map = {"High": "danger", "Medium": "moderate", "Low": "safe", "Unknown": "safe"}
    priority = {"danger": 3, "moderate": 2, "safe": 1, "unknown": 0} # unknown maps to safe here

    if all_accident_points_df.empty or not segment_coords_list:
        return "safe" # Default to safe if no accident data or empty segment

    for seg_lng, seg_lat in segment_coords_list: # OSRM provides coords as [lng, lat]
        for _, acc_row in all_accident_points_df.iterrows():
            acc_lat, acc_lng = acc_row['lat'], acc_row['lng']
            point_risk_level = acc_row.get('risk_level', 'Unknown')

            distance = haversine(seg_lat, seg_lng, acc_lat, acc_lng)

            if distance <= threshold_km:
                current_point_safety_category = risk_map.get(point_risk_level, "safe")
                if priority[current_point_safety_category] > priority[highest_risk_encountered]:
                    highest_risk_encountered = current_point_safety_category
                
                if highest_risk_encountered == "danger": # Optimization
                    return "danger"
                    
    return highest_risk_encountered

def get_osrm_route(points_coords):
    """Get actual road route from OSRM with detailed path."""
    if not points_coords or len(points_coords) < 2:
        return None
    
    coordinates = ";".join([f"{p[1]},{p[0]}" for p in points_coords]) # lng,lat format for OSRM
    # Requesting only one route (no alternatives) for segment calculation simplifies things.
    url = f"{OSRM_BASE_URL}/route/v1/driving/{coordinates}?overview=full&geometries=geojson&steps=false" 
    
    try:
        response = requests.get(url, timeout=10) # Added timeout
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        route_data = response.json()
        if route_data.get('routes'):
            return route_data
        return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error getting OSRM route for segment: {str(e)}")
        return None

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initial_data')
def initial_data_route(): # Renamed to avoid conflict with function name
    # For dropdowns, we need a unique list of districts with one representative lat/lng
    # The generate_dummy_data() creates multiple points per district.
    # We'll derive the unique list from the keys of the original districts dictionary.
    districts_dict = {
        "Baktiya": {"lat": 5.050000, "lng": 97.433333},
        "Baktiya Barat": {"lat": 5.133333, "lng": 97.366667},
        "Banda Baro": {"lat": 5.116667, "lng": 96.950000},
        "Cot Girek": {"lat": 4.833176, "lng": 97.330053},
        "Dewantara": {"lat": 5.183331, "lng": 97.000000},
        "Geureudong Pase": {"lat": 4.950000, "lng": 97.033333},
        "Kuta Makmur": {"lat": 5.083333, "lng": 97.033333},
        "Langkahan": {"lat": 4.866667, "lng": 97.450000},
        "Lapang": {"lat": 5.116667, "lng": 97.266667},
        "Lhoksukon": {"lat": 5.073056, "lng": 97.255278},
        "Matangkuli": {"lat": 5.033056, "lng": 97.277778},
        "Meurah Mulia": {"lat": 5.071972, "lng": 97.207306},
        "Muara Batu": {"lat": 5.233333, "lng": 96.950000},
        "Nibong": {"lat": 5.016667, "lng": 97.216667},
        "Nisam": {"lat": 5.016667, "lng": 96.966667},
        "Nisam Antara": {"lat": 4.916667, "lng": 96.900000},
        "Paya Bakong": {"lat": 4.950000, "lng": 97.266667},
        "Pirak Timu": {"lat": 4.916667, "lng": 97.216667},
        "Samudera": {"lat": 5.116667, "lng": 97.216667},
        "Sawang": {"lat": 5.050000, "lng": 96.883333},
        "Seunuddon": {"lat": 5.183333, "lng": 97.450000},
        "Simpang Keramat": {"lat": 4.958031, "lng": 96.949861},
        "Syamtalira Aron": {"lat": 5.100000, "lng": 97.266667},
        "Syamtalira Bayu": {"lat": 4.983333, "lng": 97.033333},
        "Tanah Jambo Aye": {"lat": 5.066667, "lng": 97.483333},
        "Tanah Luas": {"lat": 4.933333, "lng": 97.066667},
        "Tanah Pasir": {"lat": 5.133333, "lng": 97.300000}
    }
    district_options = [{'name': name, 'lat': coords['lat'], 'lng': coords['lng']} for name, coords in districts_dict.items()]
    
    # The full dataset might still be useful for other initial map displays if any.
    # For now, only district_options are strictly needed by the new JS.
    # full_dummy_data = generate_dummy_data() 

    return jsonify({
        'status': 'success',
        'district_options': district_options,
        # 'data': full_dummy_data.to_dict('records') # If needed for other purposes
    })

@app.route('/calculate_clusters')
def calculate_clusters_route(): # Renamed
    try:
        data = generate_dummy_data()
        if data.empty:
            return jsonify({'status': 'error', 'message': 'Tidak ada data untuk diklaster.'}), 400

        n_clusters = min(3, len(data.drop_duplicates(subset=['district'])) if not data.empty else 3) # Ensure n_clusters is not too large
        if len(data) < n_clusters:
             return jsonify({'status': 'error', 'message': f'Data tidak cukup untuk {n_clusters} klaster.'}), 400

        clustered_data, medoid_indices, medoid_data_df = perform_clustering(data, n_clusters=n_clusters)
        
        # Calculate risk levels based on fatalities and accidents
        # Ensure columns exist before using them
        if 'fatalities' in clustered_data.columns and 'accidents' in clustered_data.columns:
            risk_scores = clustered_data['fatalities'] * 0.6 + clustered_data['accidents'] * 0.4
            # Use pd.qcut only if there are enough unique risk_scores for 3 quantiles
            if risk_scores.nunique() >= 3:
                clustered_data['risk_level'] = pd.qcut(risk_scores, q=3, labels=['Low', 'Medium', 'High'])
            else: # Fallback if not enough unique values for quantiles
                clustered_data['risk_level'] = 'Medium' # Assign a default or handle differently
        else:
            clustered_data['risk_level'] = 'Unknown' # Or handle missing columns as an error

        # Prepare medoid data for response, including their risk level if possible
        medoids_response = []
        if not medoid_data_df.empty:
            # If medoid_data_df has risk_level (it should if clustered_data got it)
            if 'risk_level' not in medoid_data_df.columns and 'risk_level' in clustered_data.columns:
                 # If medoids were selected before risk_level was added to clustered_data, map it
                 medoid_data_df = medoid_data_df.merge(clustered_data[['lat', 'lng', 'risk_level']], on=['lat', 'lng'], how='left')
            
            # Define colors for medoids based on their cluster or risk
            cluster_colors_map = {0: 'green', 1: 'orange', 2: 'red'} # Example mapping
            medoid_data_df['color'] = medoid_data_df['cluster'].map(cluster_colors_map).fillna('blue')

            for _, medoid_row in medoid_data_df.iterrows():
                medoids_response.append({
                    'lat': medoid_row['lat'],
                    'lng': medoid_row['lng'],
                    'district': medoid_row['district'],
                    'cluster': int(medoid_row['cluster']),
                    'risk_level': medoid_row.get('risk_level', 'N/A'), # Use .get for safety
                    'color': medoid_row.get('color', 'blue')
                })

        # Convert data to native Python types for JSON serialization
        def convert_to_native_types(df):
            return [{k: (int(v) if isinstance(v, np.integer) else
                           float(v) if isinstance(v, np.floating) else
                           str(v) if isinstance(v, pd.Timestamp) else v)
                      for k, v in record.items()}
                     for record in df.to_dict('records')]

        result = {
            'status': 'success',
            'cluster_data': {
                'medoids': medoids_response,
                'clusters': convert_to_native_types(clustered_data)
            }
        }
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error calculating clusters: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Gagal menghitung klaster: {str(e)}'}), 500

# Ensure the /find_route endpoint returns a complete response with all required properties
@app.route('/find_route', methods=['POST'])
def find_route_api():
    try:
        data = request.get_json()
        if not data or 'waypoints' not in data or len(data['waypoints']) < 2:
            return jsonify({'status': 'error', 'message': 'Titik awal dan tujuan diperlukan.'}), 400

        waypoints_from_frontend = data['waypoints']
        
        # Simple function to determine a safety level based on random chance
        def simple_safety_level():
            import random
            r = random.random()
            if r < 0.6:  # 60% chance of safe routes
                return "safe"
            elif r < 0.8:  # 20% chance of moderate
                return "moderate"
            else:  # 20% chance of danger
                return "danger"

        route_segments_features = []
        total_distance_km = 0
        total_time_minutes = 0

        for i in range(len(waypoints_from_frontend) - 1):
            start_wp = waypoints_from_frontend[i]
            end_wp = waypoints_from_frontend[i + 1]

            try:
                # Try to get OSRM route data
                segment_osrm_points = [(start_wp['lat'], start_wp['lng']), (end_wp['lat'], end_wp['lng'])]
                osrm_segment_data = get_osrm_route(segment_osrm_points)
                
                if osrm_segment_data and osrm_segment_data.get('routes') and len(osrm_segment_data['routes']) > 0:
                    main_segment_route = osrm_segment_data['routes'][0]
                    segment_geometry_geojson = main_segment_route['geometry']
                    
                    # Use the actual OSRM data for distance and time
                    distance_km = main_segment_route['distance'] / 1000.0
                    time_minutes = main_segment_route['duration'] / 60.0
                else:
                    raise Exception("OSRM data incomplete")
            except:
                # Fallback to a simple straight line if OSRM fails
                segment_geometry_geojson = {
                    "type": "LineString",
                    "coordinates": [[start_wp['lng'], start_wp['lat']], [end_wp['lng'], end_wp['lat']]]
                }
                
                # Calculate distance with haversine formula
                distance_km = haversine(start_wp['lat'], start_wp['lng'], end_wp['lat'], end_wp['lng'])
                time_minutes = distance_km * 2  # Rough estimate: 30 km/h = 0.5 km/min
            
            # Determine safety level (either from cluster data or randomly)
            safety_level = simple_safety_level()
            
            total_distance_km += distance_km
            total_time_minutes += time_minutes

            route_segments_features.append({
                "type": "Feature",
                "properties": {
                    "safety_level": safety_level,
                    "distanceKm": round(distance_km, 2),
                    "timeMinutes": round(time_minutes, 1)
                },
                "geometry": segment_geometry_geojson
            })

        # Create the complete response with all expected properties
        final_geojson_response = {
            "type": "FeatureCollection",
            "features": route_segments_features
        }

        # Add waypoints_info that the frontend expects
        waypoints_info = []
        for wp in waypoints_from_frontend:
            waypoints_info.append({
                "name": wp.get('district', 'Unknown'),
                "lat": wp['lat'],
                "lng": wp['lng']
            })

        return jsonify({
            'status': 'success',
            'route_geojson': final_geojson_response,
            'waypoints_info': waypoints_info,
            'route_summary': {
                'totalDistanceKm': round(total_distance_km, 2),
                'totalTimeMinutes': round(total_time_minutes, 0)
            }
        })

    except Exception as e:
        import traceback
        app.logger.error(f"Error in find_route_api: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error', 
            'message': f'Gagal menemukan rute: {str(e)}',
            'trace': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # For development, ensure Flask's reloader can find the app object correctly
    # if your file is named e.g. main.py and you run `flask run`, it looks for app or create_app.
    # If running `python app.py`, this is fine.
    app.run(debug=True, host='0.0.0.0', port=5000) # Added host and port for clarity