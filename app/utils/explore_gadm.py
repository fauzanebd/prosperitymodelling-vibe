import json
from pathlib import Path
from collections import Counter

def explore_gadm_data():
    """
    Explore the GADM data structure to understand how to extract West Java.
    """
    base_dir = Path(__file__).parent.parent.parent
    input_file = base_dir / 'gadm41_IDN_2.json'
    
    print(f"Reading GADM data from {input_file}...")
    
    try:
        # Read the GADM GeoJSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            gadm_data = json.load(f)
        
        print(f"GeoJSON type: {gadm_data.get('type', 'Not specified')}")
        
        features = gadm_data.get('features', [])
        print(f"Total features: {len(features)}")
        
        if not features:
            print("No features found in the GeoJSON.")
            return
        
        # Sample the first feature to understand the structure
        sample_feature = features[0]
        print("\nSample feature structure:")
        print(f"  Feature type: {sample_feature.get('type', 'Not specified')}")
        
        # Explore properties
        properties = sample_feature.get('properties', {})
        print("\nAvailable properties:")
        for key, value in properties.items():
            print(f"  {key}: {value}")
        
        # Check if we can identify province names
        print("\nAnalyzing province information...")
        
        # Track what properties might contain province information
        province_prop_candidates = ['NAME_1', 'PROVINSI', 'PROVINCE', 'NAMA_PROV', 'name_1', 'GID_1']
        province_values = {}
        
        for prop in province_prop_candidates:
            values = [f.get('properties', {}).get(prop) for f in features if f.get('properties', {}).get(prop)]
            if values:
                value_counts = Counter(values)
                province_values[prop] = {
                    'total_unique': len(value_counts),
                    'samples': list(value_counts.keys())[:10]  # Show first 10 samples
                }
        
        print("\nPotential province properties:")
        for prop, data in province_values.items():
            print(f"  {prop}: {data['total_unique']} unique values")
            print(f"  Sample values: {', '.join(str(s) for s in data['samples'])}")
        
        # Specifically look for any mentions of Java/Jawa
        print("\nSearching for features related to Java/Jawa...")
        java_keywords = ['java', 'jawa', 'jabar']
        
        for i, feature in enumerate(features):
            props = feature.get('properties', {})
            
            # Check if any property contains Java-related terms
            for key, value in props.items():
                if isinstance(value, str) and any(keyword in value.lower() for keyword in java_keywords):
                    print(f"Found Java-related feature at index {i}:")
                    print(f"  {key}: {value}")
                    # Print other key properties for this feature
                    for k in ['NAME_1', 'NAME_2', 'GID_1', 'GID_2']:
                        if k in props:
                            print(f"  {k}: {props[k]}")
                    # Don't print every matching feature, just the first few
                    if i >= 5:
                        print("...")
                        break
            
            # Stop after reporting a few matches
            if i >= 5:
                break
        
    except Exception as e:
        print(f"Error exploring GADM data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore_gadm_data() 