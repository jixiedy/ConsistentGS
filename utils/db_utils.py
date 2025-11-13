import sqlite3
import os
import re
from pathlib import Path


def extract_frame_number(name):
    """Extract the frame number from an image name."""
    match = re.search(r'\d+', name)
    return int(match.group()) if match else -1


def find_colmap_database(source_path, is_custom_dof=False):
    """Find the COLMAP database file in the source path."""
    # Try common locations - prioritize root directory which is COLMAP's default location
    common_locations = [
        # Root directory (most common location)
        os.path.join(source_path, "database.db"),
        # Custom DOF database name
        os.path.join(source_path, "dof_rename.db") if is_custom_dof else None,
        # Other possible locations
        os.path.join(source_path, "sparse_dof", "database.db") if is_custom_dof else os.path.join(source_path, "sparse", "database.db"),
        os.path.join(source_path, "sparse_dof/0", "database.db") if is_custom_dof else os.path.join(source_path, "sparse/0", "database.db"),
    ]
    
    for location in common_locations:
        if location and os.path.exists(location):
            print(f"Found COLMAP database at {location}")
            return location
    
    print(f"Could not find COLMAP database in {source_path}")
    return None


def read_colmap_adjacency(db_path):
    """
    Read image adjacency information from a COLMAP database.
    
    Returns:
        A dictionary mapping each image name to a list of adjacent image names,
        sorted by match count (highest first).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all images
    cursor.execute("SELECT image_id, name FROM images")
    image_id_to_name = {row[0]: row[1] for row in cursor.fetchall()}
    
    print(f"Found {len(image_id_to_name)} images in the COLMAP database")
    if image_id_to_name:
        print(f"Example image names: {list(image_id_to_name.values())[:5]}")
    
    # Get all matches
    cursor.execute("SELECT pair_id, rows FROM matches")
    matches = {row[0]: row[1] for row in cursor.fetchall()}
    
    print(f"Found {len(matches)} match pairs in the COLMAP database")
    
    adjacency = {}
    
    for pair_id, match_count in matches.items():
        # Extract the two image IDs from the pair_id
        image_id1 = pair_id % 2147483647
        image_id2 = pair_id // 2147483647
        
        if image_id1 in image_id_to_name and image_id2 in image_id_to_name:
            image_name1 = image_id_to_name[image_id1]
            image_name2 = image_id_to_name[image_id2]
            
            # Store the adjacency information
            if image_name1 not in adjacency:
                adjacency[image_name1] = []
            if image_name2 not in adjacency:
                adjacency[image_name2] = []
            
            adjacency[image_name1].append((image_name2, match_count))
            adjacency[image_name2].append((image_name1, match_count))
    
    # Sort adjacent images by match count (highest first)
    for image_name in adjacency:
        adjacency[image_name].sort(key=lambda x: x[1], reverse=True)
    
    conn.close()
    
    return adjacency