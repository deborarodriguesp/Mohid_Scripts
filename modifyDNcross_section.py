import os
import re

def modify_drainage_network(input_path, output_path, d_height=0.0, d_top_width=0.0, d_bottom_width=0.0):
    """
    Modifies the geometric properties of cross-section nodes in a MOHID Land .dnt file.
    Positive delta values add to the dimensions, negative values subtract.
    Automatically recalculates the BOTTOM_LEVEL for each node based on its TERRAIN_LEVEL.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    in_node = False
    node_data = {}

    # Regular expression to capture "KEY  : VALUE" pairs
    kv_pattern = re.compile(r'^\s*([A-Za-z0-9_]+)\s*:\s*(.*)$')

    for line in lines:
        stripped = line.strip()
        
        if stripped == "<BeginNode>":
            in_node = True
            new_lines.append(line)
            node_data = {}
            continue
        
        if stripped == "<EndNode>":
            # --- APPLY DELTAS AND RECALCULATE GEOMETRY ---
            # Fallback to 0 if a key is missing for any reason
            h = float(node_data.get('HEIGHT', 0)) + d_height
            tw = float(node_data.get('TOP_WIDTH', 0)) + d_top_width
            bw = float(node_data.get('BOTTOM_WIDTH', 0)) + d_bottom_width

            # Prevent dimensions from becoming negative
            if h < 0: h = 0.0
            if tw < 0: tw = 0.0
            if bw < 0: bw = 0.0
            
            bottom_level = -h
            
            # Write structural keys that remain unchanged in their original order
            node_keys_order = [
                'ID', 'COORDINATES', 'GRID_I', 'GRID_J', 
                'CROSS_SECTION_TYPE', 'CROSS_SECTION_ORIGIN', 
                'DRAINED_AREA', 'TERRAIN_LEVEL'
            ]
            
            for key in node_keys_order:
                if key in node_data:
                    new_lines.append(f"{key:<26}: {node_data[key]}\n")
            
            # Append modified and recalculated variables with correct text column padding
            new_lines.append(f"{'BOTTOM_LEVEL':<26}: {bottom_level:.7f}\n")
            new_lines.append(f"{'BOTTOM_WIDTH':<26}: {bw:.7f}\n")
            new_lines.append(f"{'TOP_WIDTH':<26}: {tw:.7f}\n")
            new_lines.append(f"{'HEIGHT':<26}: {h:.7f}\n")
            
            new_lines.append(line)  # Appends the </EndNode> closing tag
            in_node = False
            continue

        if in_node:
            match = kv_pattern.match(line)
            if match:
                key, val = match.groups()
                node_data[key.strip()] = val.strip()
        else:
            # Keep global file headers/projections intact
            new_lines.append(line)

    # Save the newly generated network file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Success! Modified file saved to:\n--> {output_path}")

# ============================================================
# CONFIGURATION
# ============================================================
file_path = r"D:\DOUTORAMENTO\TwinStream\Projetos\TwinStream_ModelosHidrologicos\Modelos\Md_Cor_Rvr_InilWtrDepth0\GeneralData\DigitalTerrain"
original = "DrainageNetwork.dnt"
modified = "DrainageNetwork_Mod.dnt"

original_file = os.path.join(file_path, original)
modified_file = os.path.join(file_path, modified)

# Define the delta adjustment values (in meters) for all cross sections:
delta_values = {
    "d_height": 1.0,       # Adds x meters to the depth/height
    "d_top_width": 0,      # Widens the channel top by x meters
    "d_bottom_width": 0    # Widens the channel bottom by x meters
}

# Execute the modifier function
modify_drainage_network(
    input_path=original_file,
    output_path=modified_file,
    **delta_values
)