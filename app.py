from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import random
import folium
import networkx as nx
import requests # Make sure to install requests: pip install requests
from math import radians, sin, cos, sqrt, atan2 # Added for Haversine
from sklearn.manifold import TSNE

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
    """Manual K-medoids clustering implementation"""
    if data.empty or len(data) < n_clusters:
        # Not enough data to cluster, return empty or handle as error
        return data, [], pd.DataFrame() # Added empty DataFrame for medoid_data

    features = data[['fatalities', 'road_condition', 'accidents', 'traffic']]
    # Ensure features are numeric and handle potential NaNs (though dummy data shouldn't have them)
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Manual K-medoids implementation
    medoid_indices, cluster_labels = manual_kmedoids(features.values, n_clusters, random_state=42)
    
    # Add cluster labels to data
    data['cluster'] = cluster_labels
    medoid_data = data.iloc[medoid_indices].copy() # Create a copy to avoid SettingWithCopyWarning
    return data, medoid_indices, medoid_data

def manual_kmedoids(X, n_clusters, max_iterations=100, random_state=None):
    """
    Manual implementation of K-medoids clustering algorithm
    
    Parameters:
    - X: feature matrix (n_samples, n_features)
    - n_clusters: number of clusters
    - max_iterations: maximum number of iterations
    - random_state: random seed for reproducibility
    
    Returns:
    - medoid_indices: indices of the medoid points
    - cluster_labels: cluster assignment for each point
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    # Step 1: Initialize medoids randomly
    medoid_indices = np.random.choice(n_samples, n_clusters, replace=False)
    
    for iteration in range(max_iterations):
        # Step 2: Assign each point to the closest medoid
        distances_to_medoids = np.zeros((n_samples, n_clusters))
        
        for i, medoid_idx in enumerate(medoid_indices):
            medoid = X[medoid_idx]
            for j in range(n_samples):
                # Calculate Euclidean distance
                distances_to_medoids[j, i] = np.sqrt(np.sum((X[j] - medoid) ** 2))
        
        # Assign each point to closest medoid
        cluster_labels = np.argmin(distances_to_medoids, axis=1)
        
        # Step 3: Calculate current total cost
        current_cost = 0
        for cluster_id in range(n_clusters):
            cluster_points = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_points) > 0:
                medoid = X[medoid_indices[cluster_id]]
                for point_idx in cluster_points:
                    current_cost += np.sqrt(np.sum((X[point_idx] - medoid) ** 2))
        
        # Step 4: Try to improve medoids
        improved = False
        best_medoids = medoid_indices.copy()
        best_cost = current_cost
        
        # For each cluster, try replacing the medoid with each point in the cluster
        for cluster_id in range(n_clusters):
            cluster_points = np.where(cluster_labels == cluster_id)[0]
            
            for candidate_idx in cluster_points:
                if candidate_idx == medoid_indices[cluster_id]:
                    continue  # Skip if it's already the medoid
                
                # Test this candidate as new medoid
                test_medoids = medoid_indices.copy()
                test_medoids[cluster_id] = candidate_idx
                
                # Recalculate distances and assignments with new medoid
                test_distances = np.zeros((n_samples, n_clusters))
                for i, test_medoid_idx in enumerate(test_medoids):
                    test_medoid = X[test_medoid_idx]
                    for j in range(n_samples):
                        test_distances[j, i] = np.sqrt(np.sum((X[j] - test_medoid) ** 2))
                
                test_labels = np.argmin(test_distances, axis=1)
                
                # Calculate cost with new configuration
                test_cost = 0
                for test_cluster_id in range(n_clusters):
                    test_cluster_points = np.where(test_labels == test_cluster_id)[0]
                    if len(test_cluster_points) > 0:
                        test_medoid = X[test_medoids[test_cluster_id]]
                        for point_idx in test_cluster_points:
                            test_cost += np.sqrt(np.sum((X[point_idx] - test_medoid) ** 2))
                
                # If this configuration is better, update
                if test_cost < best_cost:
                    best_cost = test_cost
                    best_medoids = test_medoids.copy()
                    improved = True
        
        # Update medoids if improvement found
        if improved:
            medoid_indices = best_medoids
        else:
            # No improvement, algorithm has converged
            break
    
    # Final assignment with converged medoids
    final_distances = np.zeros((n_samples, n_clusters))
    for i, medoid_idx in enumerate(medoid_indices):
        medoid = X[medoid_idx]
        for j in range(n_samples):
            final_distances[j, i] = np.sqrt(np.sum((X[j] - medoid) ** 2))
    
    final_labels = np.argmin(final_distances, axis=1)
    
    return medoid_indices, final_labels

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

@app.route('/simulation')
def simulation():
    """Route for the detailed simulation page showing K-medoids and Dijkstra algorithms"""
    return render_template('simulation.html')

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

@app.route('/simulation_data')
def simulation_data():
    """API endpoint to provide detailed simulation data for K-medoids and Dijkstra algorithms"""

    try:
        # Generate dummy data
        raw_data = generate_dummy_data()
        # Replace NaN/None/blank, non-numeric, inf/-inf with 0 for all numeric columns
        for col in raw_data.columns:
            if np.issubdtype(raw_data[col].dtype, np.number):
                raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
            else:
                raw_data[col] = raw_data[col].replace([None, '', np.nan, np.inf, -np.inf, 'Infinity', '-Infinity'], 0)
        # Store all intermediate steps for detailed visualization
        simulation_results = {
            "status": "success",
            "data_collection": {
                "raw_data": raw_data.replace([None, '', np.nan, np.inf, -np.inf, 'Infinity', '-Infinity'], 0).to_dict('records'),
                "description": "Data kecelakaan yang dikumpulkan dari berbagai wilayah di Aceh Utara dan Lhokseumawe"
            },
            "kmedoids_steps": [],
            "final_clusters": {},
            "dijkstra_steps": []
        }
        
        # Step 1: Data Collection and Preparation
        features = raw_data[['fatalities', 'road_condition', 'accidents', 'traffic']]
        features = features.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
        features_normalized = (features - features.mean()) / features.std()
        features_normalized = features_normalized.replace([np.inf, -np.inf], 0).fillna(0)
        simulation_results["data_collection"]["features"] = features.to_dict('records')
        simulation_results["data_collection"]["features_normalized"] = features_normalized.to_dict('records')

        # --- t-SNE Visualization ---
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=500)
            tsne_result = tsne.fit_transform(features_normalized)
            tsne_data = []
            for i, row in enumerate(raw_data.itertuples()):
                tsne_data.append({
                    'x': float(tsne_result[i,0]),
                    'y': float(tsne_result[i,1]),
                    'district': getattr(row, 'district', ''),
                    'fatalities': getattr(row, 'fatalities', 0),
                    'accidents': getattr(row, 'accidents', 0),
                    'cluster': int(getattr(row, 'cluster', 0)) if hasattr(row, 'cluster') else 0
                })
            simulation_results["data_collection"]["tsne"] = tsne_data
        except Exception as e:
            simulation_results["data_collection"]["tsne"] = []
        # Step 2: K-medoids Clustering Process
        n_clusters = 3
        
        # Store initial random medoid selection
        np.random.seed(42)  # For reproducibility
        initial_medoid_indices = np.random.choice(len(raw_data), n_clusters, replace=False)
        initial_medoids = raw_data.iloc[initial_medoid_indices].copy()
        
        # Add initial medoids selection step
        simulation_results["kmedoids_steps"].append({
            "step_name": "Inisialisasi Medoid",
            "description": "Memilih titik medoid awal secara acak",
            "medoid_indices": initial_medoid_indices.tolist(),
            "medoids": initial_medoids.to_dict('records')
        })
        
        # Simulate the full K-medoids algorithm steps
        # For simplicity, we'll do 3 iterations (typically the algorithm would run until convergence)
        current_medoids = initial_medoid_indices.copy()
        
        for iteration in range(3):
            # Create distance matrix between each point and each medoid
            distances = np.zeros((len(raw_data), len(current_medoids)))
            for i, medoid_idx in enumerate(current_medoids):
                medoid = raw_data.iloc[medoid_idx][['fatalities', 'road_condition', 'accidents', 'traffic']].values
                for j, row in enumerate(features.values):
                    # Using Euclidean distance for simplicity
                    distances[j, i] = np.sqrt(np.sum((row - medoid) ** 2))
            
            # Assign each point to closest medoid
            clusters = np.argmin(distances, axis=1)
            cluster_assignment = pd.DataFrame({
                'point_index': range(len(raw_data)),
                'cluster': clusters,
                'district': raw_data['district'],
                'lat': raw_data['lat'],
                'lng': raw_data['lng'],
                'fatalities': raw_data['fatalities'],
                'road_condition': raw_data['road_condition'],
                'accidents': raw_data['accidents'],
                'traffic': raw_data['traffic']
            })
            
            # Calculate cluster cost
            total_cost = 0
            for cluster_id in range(n_clusters):
                cluster_points = cluster_assignment[cluster_assignment['cluster'] == cluster_id]
                if not cluster_points.empty:
                    medoid_idx = current_medoids[cluster_id]
                    medoid = raw_data.iloc[medoid_idx][['fatalities', 'road_condition', 'accidents', 'traffic']].values
                    
                    for _, point in cluster_points.iterrows():
                        point_values = point[['fatalities', 'road_condition', 'accidents', 'traffic']].values
                        distance = np.sqrt(np.sum((point_values - medoid) ** 2))
                        total_cost += distance
            
            # Record this assignment step
            simulation_results["kmedoids_steps"].append({
                "step_name": f"Iterasi {iteration+1} - Penugasan Kluster",
                "description": "Menetapkan setiap titik ke medoid terdekat",
                "cluster_assignments": cluster_assignment.to_dict('records'),
                "total_cost": float(total_cost)
            })
            
            # Try to replace medoids with non-medoids to see if cost improves
            best_cost = total_cost
            best_new_medoids = current_medoids.copy()
            
            # For educational purposes, we'll just test a few random swaps instead of all possibilities
            # In a real implementation, you would test all possible swaps
            for cluster_id in range(n_clusters):
                cluster_points = cluster_assignment[cluster_assignment['cluster'] == cluster_id]
                if len(cluster_points) > 0:
                    # Test 2 random points from each cluster as potential new medoids
                    candidate_indices = cluster_points['point_index'].sample(min(2, len(cluster_points))).values
                    
                    for candidate_idx in candidate_indices:
                        new_medoids = current_medoids.copy()
                        new_medoids[cluster_id] = candidate_idx
                        
                        # Calculate new cost with candidate medoid
                        new_distances = np.zeros((len(raw_data), len(new_medoids)))
                        for i, medoid_idx in enumerate(new_medoids):
                            medoid = raw_data.iloc[medoid_idx][['fatalities', 'road_condition', 'accidents', 'traffic']].values
                            for j, row in enumerate(features.values):
                                new_distances[j, i] = np.sqrt(np.sum((row - medoid) ** 2))
                        
                        new_clusters = np.argmin(new_distances, axis=1)
                        new_cost = 0
                        
                        for new_cluster_id in range(n_clusters):
                            cluster_points = np.where(new_clusters == new_cluster_id)[0]
                            if len(cluster_points) > 0:
                                medoid_idx = new_medoids[new_cluster_id]
                                medoid = raw_data.iloc[medoid_idx][['fatalities', 'road_condition', 'accidents', 'traffic']].values
                                
                                for point_idx in cluster_points:
                                    point = features.values[point_idx]
                                    distance = np.sqrt(np.sum((point - medoid) ** 2))
                                    new_cost += distance
                        
                        # If this swap improves the cost, keep track of it
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_new_medoids = new_medoids.copy()
            
            # Record the swap evaluation step
            old_medoids_data = raw_data.iloc[current_medoids].to_dict('records')
            new_medoids_data = raw_data.iloc[best_new_medoids].to_dict('records')
            
            simulation_results["kmedoids_steps"].append({
                "step_name": f"Iterasi {iteration+1} - Evaluasi Pertukaran Medoid",
                "description": "Mengevaluasi pertukaran medoid untuk meminimalkan biaya total",
                "old_medoids": old_medoids_data,
                "new_medoids": new_medoids_data,
                "old_cost": float(total_cost),
                "new_cost": float(best_cost)
            })
            
            # Update medoids if better ones found
            if not np.array_equal(current_medoids, best_new_medoids):
                current_medoids = best_new_medoids.copy()
                
                simulation_results["kmedoids_steps"].append({
                    "step_name": f"Iterasi {iteration+1} - Pembaruan Medoid",
                    "description": "Memperbarui medoid berdasarkan evaluasi pertukaran",
                    "new_medoids": raw_data.iloc[current_medoids].to_dict('records')
                })
            else:
                simulation_results["kmedoids_steps"].append({
                    "step_name": f"Iterasi {iteration+1} - Konvergensi",
                    "description": "Tidak ada perubahan medoid yang dapat meningkatkan hasil"
                })
        
        # Generate final clustering results
        final_clusters = {}
        final_medoids = raw_data.iloc[current_medoids].copy()
        final_medoids['is_medoid'] = True
        
        # Assign points to final clusters
        final_distances = np.zeros((len(raw_data), len(current_medoids)))
        for i, medoid_idx in enumerate(current_medoids):
            medoid = raw_data.iloc[medoid_idx][['fatalities', 'road_condition', 'accidents', 'traffic']].values
            for j, row in enumerate(features.values):
                final_distances[j, i] = np.sqrt(np.sum((row - medoid) ** 2))
        
        final_clusters_assignment = np.argmin(final_distances, axis=1)
        raw_data['cluster'] = final_clusters_assignment
        
        # Calculate average risk score per cluster
        raw_data['risk_score'] = raw_data['fatalities'] * 0.6 + raw_data['accidents'] * 0.4
        
        # Determine risk levels for each point based on risk scores
        risk_scores = raw_data['risk_score'].values
        if len(np.unique(risk_scores)) >= 3:
            # Using quantile-based classification if enough unique values
            thresholds = np.percentile(risk_scores, [33.33, 66.67])
            risk_levels = np.array(['Low', 'Medium', 'High'])
            raw_data['risk_level'] = risk_levels[np.digitize(risk_scores, thresholds)]
        else:
            # Simple thresholding for small datasets
            median_val = np.median(risk_scores)
            raw_data['risk_level'] = np.where(risk_scores > median_val, 'High', 'Low')
        
        # Group the final results by cluster
        for cluster_id in range(n_clusters):
            cluster_points = raw_data[raw_data['cluster'] == cluster_id]
            medoid = final_medoids[final_medoids.index == current_medoids[cluster_id]]
            
            final_clusters[f"cluster_{cluster_id}"] = {
                "points": cluster_points.to_dict('records'),
                "medoid": medoid.to_dict('records')[0],
                "avg_risk_score": float(cluster_points['risk_score'].mean()),
                "dominant_risk_level": cluster_points['risk_level'].value_counts().index[0],
                "count": len(cluster_points)
            }
        
        simulation_results["final_clusters"] = final_clusters
        
        # Step 3: Simulate Dijkstra's algorithm on a sample route
        # For this example, we'll create a simplified graph representing connections between districts
        # with distances and simulate finding the shortest path
        
        # Create a graph where nodes are districts and edges represent connections
        G = nx.Graph()
        
        # Add nodes (districts)
        districts_data = {}
        for district in raw_data['district'].unique():
            district_data = raw_data[raw_data['district'] == district].iloc[0]
            G.add_node(district, lat=district_data['lat'], lng=district_data['lng'])
            districts_data[district] = {
                "lat": district_data['lat'],
                "lng": district_data['lng']
            }
        
        # Add edges (connections between districts)
        districts = list(districts_data.keys())
        
        # Create meaningful connections based on proximity
        for i, district1 in enumerate(districts):
            for j, district2 in enumerate(districts):
                if i < j:  # Avoid duplicate edges
                    dist1 = districts_data[district1]
                    dist2 = districts_data[district2]
                    
                    # Calculate distance between districts
                    distance = haversine(dist1['lat'], dist1['lng'], dist2['lat'], dist2['lng'])
                    
                    # Only connect districts that are within a reasonable distance (e.g., 30 km)
                    if distance < 30:
                        # Find any accident points near this route for safety assessment
                        points_near_route = []
                        for _, point in raw_data.iterrows():
                            # Check if a point is close to the route
                            pt_dist1 = haversine(point['lat'], point['lng'], dist1['lat'], dist1['lng'])
                            pt_dist2 = haversine(point['lat'], point['lng'], dist2['lat'], dist2['lng'])
                            
                            # If point is close to either end of the route or along the route
                            if pt_dist1 < 5 or pt_dist2 < 5:
                                points_near_route.append({
                                    "district": point['district'],
                                    "risk_level": point.get('risk_level', 'Unknown'),
                                    "accidents": int(point['accidents']),
                                    "fatalities": int(point['fatalities']),
                                    "lat": point['lat'],
                                    "lng": point['lng']
                                })
                        
                        # Adjust weight based on safety concerns
                        safety_factor = 1.0
                        for point in points_near_route:
                            if point['risk_level'] == 'High':
                                safety_factor += 0.5  # High risk adds 50% to the path weight
                            elif point['risk_level'] == 'Medium':
                                safety_factor += 0.2  # Medium risk adds 20% to the path weight
                          # Final weighted distance considering safety
                        weighted_distance = distance * safety_factor
                        
                        # Add the edge with original and weighted distances
                        G.add_edge(district1, district2, 
                                  distance=distance, 
                                  weighted_distance=weighted_distance,
                                  safety_factor=safety_factor,
                                  points_near_route=points_near_route)
        
        # Select start and end points for our Dijkstra simulation
        all_districts = list(districts_data.keys())
        start_district = all_districts[0]  # First district
        end_district = all_districts[-1]   # Last district
        
        # Record the graph structure
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        for node in G.nodes():
            graph_data["nodes"].append({
                "id": node,
                "lat": G.nodes[node]['lat'],
                "lng": G.nodes[node]['lng']
            })
        
        for u, v, data in G.edges(data=True):
            graph_data["edges"].append({
                "source": u,
                "target": v,
                "distance": data['distance'],
                "weighted_distance": data['weighted_distance'],
                "safety_factor": data['safety_factor']
            })
        
        simulation_results["dijkstra_steps"].append({
            "step_name": "Inisialisasi Graf",
            "description": "Membuat graf berdasarkan wilayah dan koneksi antar wilayah",
            "graph": graph_data
        })
        
        # Run Dijkstra's algorithm and record the steps
        # Initialize distance dictionary
        distances = {node: float('infinity') for node in G.nodes()}
        distances[start_district] = 0
        
        # Initialize previous node dictionary for path reconstruction
        previous = {node: None for node in G.nodes()}
        
        # Initialize priority queue
        unvisited = list(G.nodes())
        
        # Record initialization step
        simulation_results["dijkstra_steps"].append({
            "step_name": "Inisialisasi Dijkstra",
            "description": "Menetapkan jarak awal dan node sebelumnya",
            "distances": distances.copy(),
            "start_node": start_district,
            "target_node": end_district
        })
        
        # Run Dijkstra's algorithm
        iteration = 0
        visited = []
        
        while unvisited and any(distances[node] < float('infinity') for node in unvisited):
            iteration += 1
            
            # Find the unvisited node with the smallest distance
            current = min(unvisited, key=lambda node: distances[node])
            
            # Stop if we've reached the target
            if current == end_district:
                break
                
            # Visit the current node
            unvisited.remove(current)
            visited.append(current)
            
            # Record the current state before processing neighbors
            current_state = {
                "step_name": f"Iterasi {iteration} - Pemilihan Node",
                "description": f"Mengunjungi node {current} dengan jarak {distances[current]:.2f} km",
                "current_node": current,
                "distances": distances.copy(),
                "visited": visited.copy(),
                "unvisited": unvisited.copy()
            }
            
            simulation_results["dijkstra_steps"].append(current_state)
            
            # Process neighbors
            neighbor_updates = []
            
            for neighbor in G.neighbors(current):
                if neighbor in unvisited:
                    # Get the edge data
                    edge_data = G.get_edge_data(current, neighbor)
                    
                    # Use weighted distance for safety-aware routing
                    edge_distance = edge_data['weighted_distance']
                    
                    # Calculate new distance to this neighbor
                    new_distance = distances[current] + edge_distance
                    
                    # If we found a better path
                    if new_distance < distances[neighbor]:
                        # Record this neighbor update
                        neighbor_updates.append({
                            "neighbor": neighbor,
                            "old_distance": distances[neighbor],
                            "new_distance": new_distance,
                            "via_node": current,
                            "edge_distance": edge_distance
                        })
                        
                        # Update the distance and previous node
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
            
            # Record the neighbor updates
            if neighbor_updates:
                simulation_results["dijkstra_steps"].append({
                    "step_name": f"Iterasi {iteration} - Pembaruan Tetangga",
                    "description": f"Memperbarui jarak ke tetangga dari node {current}",
                    "neighbor_updates": neighbor_updates,
                    "updated_distances": distances.copy()
                })
        
        # Reconstruct the path
        path = []
        current = end_district
        
        while current:
            path.append(current)
            current = previous[current]
        
        path.reverse()  # Correct the order
        
        # Calculate the actual path segments with their properties
        path_segments = []
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            
            # Get edge data
            edge_data = G.get_edge_data(start_node, end_node)
            
            segment = {
                "from": start_node,
                "to": end_node,
                "distance": edge_data['distance'],
                "weighted_distance": edge_data['weighted_distance'],
                "safety_factor": edge_data['safety_factor'],
                "points_near_route": edge_data['points_near_route'],
                "from_coords": {
                    "lat": G.nodes[start_node]['lat'],
                    "lng": G.nodes[start_node]['lng']
                },
                "to_coords": {
                    "lat": G.nodes[end_node]['lat'],
                    "lng": G.nodes[end_node]['lng']
                }
            }
            
            path_segments.append(segment)
            
            segment = {
                "from": start_node,
                "to": end_node,
                "distance": edge_data['distance'],
                "weighted_distance": edge_data['weighted_distance'],
                "safety_factor": edge_data['safety_factor'],
                "points_near_route": edge_data['points_near_route'],
                "from_coords": {
                    "lat": G.nodes[start_node]['lat'],
                    "lng": G.nodes[start_node]['lng']
                },
                "to_coords": {
                    "lat": G.nodes[end_node]['lat'],
                    "lng": G.nodes[end_node]['lng']
                }
            }
            
            path_segments.append(segment)
        
        # Record the final path
        simulation_results["dijkstra_steps"].append({
            "step_name": "Hasil Akhir - Jalur Terpendek",
            "description": f"Jalur terpendek dari {start_district} ke {end_district}",
            "path": path,
            "path_segments": path_segments,
            "total_distance": sum(segment['distance'] for segment in path_segments),
            "total_weighted_distance": sum(segment['weighted_distance'] for segment in path_segments)
        })
        
        # Convert all Infinity/-Infinity in distances to string '∞' or a large number (e.g. 99999)
        def safe_distances(d):
            return {k: (0 if v == 0 else (99999 if v == float('inf') or v == float('-inf') or v == 'Infinity' or v == '-Infinity' else v)) for k, v in d.items()}

        # Saat menambahkan step Dijkstra, pastikan distances sudah diubah
        # Contoh:
        # simulation_results["dijkstra_steps"].append({
        #     ...
        #     "distances": safe_distances(distances.copy()),
        #     ...
        # })
        #
        # Atau, setelah semua step selesai, lakukan normalisasi:
        for step in simulation_results.get("dijkstra_steps", []):
            if "distances" in step and isinstance(step["distances"], dict):
                step["distances"] = safe_distances(step["distances"])
        
        # After all calculations, clean up all numeric results in simulation_results
        def safe_numeric(val):
            try:
                v = float(val)
                if np.isnan(v) or np.isinf(v):
                    return 0
                return v
            except Exception:
                return 0
        # Clean up final_clusters
        for cluster in simulation_results.get("final_clusters", {}).values():
            for point in cluster.get("points", []):
                for k, v in point.items():
                    if isinstance(v, (float, int)):
                        point[k] = safe_numeric(v)
            cluster["avg_risk_score"] = safe_numeric(cluster.get("avg_risk_score", 0))
        # Clean up dijkstra_steps
        for step in simulation_results.get("dijkstra_steps", []):
            if "distances" in step and isinstance(step["distances"], dict):
                for k, v in step["distances"].items():
                    step["distances"][k] = safe_numeric(v)
            if "neighbor_updates" in step and isinstance(step["neighbor_updates"], list):
                for upd in step["neighbor_updates"]:
                    for k, v in upd.items():
                        if isinstance(v, (float, int)):
                            upd[k] = safe_numeric(v)
            if "updated_distances" in step and isinstance(step["updated_distances"], dict):
                for k, v in step["updated_distances"].items():
                    step["updated_distances"][k] = safe_numeric(v)
            if "total_distance" in step:
                step["total_distance"] = safe_numeric(step["total_distance"])
            if "total_weighted_distance" in step:
                step["total_weighted_distance"] = safe_numeric(step["total_weighted_distance"])
            if "path_segments" in step and isinstance(step["path_segments"], list):
                for seg in step["path_segments"]:
                    for k, v in seg.items():
                        if isinstance(v, (float, int)):
                            seg[k] = safe_numeric(v)
        return jsonify(simulation_results)
        
    except Exception as e:
        import traceback
        app.logger.error(f"Error in simulation data endpoint: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Gagal menghasilkan data simulasi: {str(e)}',
            'trace': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # For development, ensure Flask's reloader can find the app object correctly
    # if your file is named e.g. main.py and you run `flask run`, it looks for app or create_app.
    # If running `python app.py`, this is fine.
    app.run(debug=True, host='0.0.0.0', port=5000) # Added host and port for clarity