import json
import os
from pathlib import Path

def extract_west_java():
    """
    Extract West Java (Jawa Barat) features from the GADM data file
    and save as a smaller GeoJSON file.
    """
    # Define file paths
    base_dir = Path(__file__).parent.parent.parent
    input_file = base_dir / 'gadm41_IDN_2.json'
    output_dir = base_dir / 'app' / 'static' / 'data'
    output_file = output_dir / 'west_java.geojson'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading GADM data from {input_file}...")
    
    try:
        # Read the GADM GeoJSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            gadm_data = json.load(f)
        
        # Based on exploration, West Java is "JawaBarat" in NAME_1 property
        west_java_name = "JawaBarat"
        
        # Extract features for West Java only
        west_java_features = []
        total_features = len(gadm_data['features'])
        
        print(f"Searching through {total_features} features for '{west_java_name}'...")
        
        # Filter features for West Java
        for feature in gadm_data['features']:
            properties = feature.get('properties', {})
            
            # Check if this feature is in West Java province
            if properties.get('NAME_1') == west_java_name:
                # Add a more user-friendly name for display
                if 'NAME_2' in properties:
                    # Get the district/city name
                    district_name = properties['NAME_2']
                    # Add spaces between words (JawaBarat -> Jawa Barat)
                    district_name = ''.join([' ' + c if c.isupper() and i > 0 else c 
                                            for i, c in enumerate(district_name)]).strip()
                    properties['name'] = district_name
                
                # Add the feature to our filtered collection
                west_java_features.append(feature)
                print(f"Found West Java region: {properties.get('NAME_2', 'Unknown')}")
        
        # Create a new GeoJSON with just West Java features
        west_java_geojson = {
            'type': 'FeatureCollection',
            'features': west_java_features
        }
        
        # Save the filtered GeoJSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(west_java_geojson, f)
        
        print(f"Extracted {len(west_java_features)} West Java features and saved to {output_file}")
        print(f"Original file size: {os.path.getsize(input_file) / (1024*1024):.2f} MB")
        print(f"West Java file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        return str(output_file)
    
    except Exception as e:
        print(f"Error processing GADM data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    extract_west_java() 