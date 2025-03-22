import geopandas as gpd
import os
import json
from pathlib import Path

def process_shapefile():
    """
    Process the large Indonesia shapefile to extract only West Java regions.
    Saves the result as a GeoJSON file for web use.
    """
    # Get the base directory
    base_dir = Path(__file__).parent.parent.parent.parent
    
    # Path to the shapefile
    shapefile_path = base_dir / 'BATAS KABUPATEN KOTA DESEMBER 2019 DUKCAPIL.shp'
    
    # Path where the output GeoJSON will be saved
    output_path = base_dir / 'app' / 'static' / 'data'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read the shapefile
    print(f"Reading shapefile from {shapefile_path}...")
    try:
        gdf = gpd.read_file(str(shapefile_path))
        print(f"Successfully read shapefile with {len(gdf)} features")
        
        # Print the columns to understand the data structure
        print(f"Columns in the shapefile: {gdf.columns.tolist()}")
        
        # Attempt to filter by province - adapt based on actual column names
        province_column = None
        for col in gdf.columns:
            if col.upper() in ['PROVINSI', 'NAMA_PROV', 'PROVINCE', 'PROVINSI_', 'KD_PROV']:
                province_column = col
                break
                
        if not province_column:
            # If no standard column name is found, examine data to find the province column
            for col in gdf.columns:
                if gdf[col].dtype == 'object':  # Only check string columns
                    unique_vals = set(str(x).upper() for x in gdf[col].dropna().unique())
                    if any('JAWA BARAT' in x for x in unique_vals) or any('WEST JAVA' in x for x in unique_vals):
                        province_column = col
                        print(f"Found potential province column: {col}")
                        break
            
        if province_column:
            # Check all values in this column to find matches for "Jawa Barat"
            unique_provinces = set(str(x).upper() for x in gdf[province_column].dropna().unique())
            print(f"Unique values in {province_column}: {unique_provinces}")
            
            # Try various ways to match West Java
            west_java_matches = [p for p in unique_provinces if 'JAWA BARAT' in p or 'WEST JAVA' in p or 'JABAR' in p]
            if west_java_matches:
                west_java_name = west_java_matches[0]
                print(f"Found West Java as: {west_java_name}")
                
                # Filter for West Java
                west_java = gdf[gdf[province_column].astype(str).str.upper() == west_java_name]
                print(f"Found {len(west_java)} regions in West Java")
            else:
                print(f"Could not find West Java in column {province_column}")
                # As a fallback, just save a sample of the data for examination
                sample_gdf = gdf.head(5)
                sample_output = output_path / 'sample_indonesia.geojson'
                sample_gdf.to_file(sample_output, driver='GeoJSON')
                print(f"Saved a sample of 5 regions to {sample_output} for examination")
                return None
        else:
            print("Could not identify a province column. Saving full data for examination.")
            # Save full data for examination
            gdf.to_file(output_path / 'full_indonesia.geojson', driver='GeoJSON')
            return None
            
        if len(west_java) == 0:
            print("No West Java regions found. Please examine the data structure.")
            return None
            
        # Simplify geometry to reduce file size (adjust tolerance as needed)
        west_java['geometry'] = west_java['geometry'].simplify(tolerance=0.001)
        
        # Save as GeoJSON
        output_file = output_path / 'west_java.geojson'
        west_java.to_file(str(output_file), driver='GeoJSON')
        print(f"Successfully saved West Java shapefile to {output_file}")
        
        return str(output_file)
        
    except Exception as e:
        print(f"Error processing shapefile: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    process_shapefile() 