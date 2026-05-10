def format_boxes_for_comet(gt_boxes):
    """
    Converts Ground Truth box data (center to top-left) 
    to fit the Comet ML annotation schema.
    """
    annotations = []
    
    # Map Waymo object types to readable strings
    type_map = {
        1: "Vehicle",
        2: "Pedestrian",
        3: "Sign",
        4: "Cyclist"
    }

    for box in gt_boxes:
        top_left_x = box["center_x"] - (box["length"] / 2.0)
        top_left_y = box["center_y"] - (box["width"] / 2.0)
        
        class_name = type_map.get(box["type"], "Unknown")
        short_id = box["id"][:6]
        
        annotations.append({
            "name": class_name,
            "data": [{
                "label": f"ID:{short_id}",
                "boxes": [[top_left_x, top_left_y, box["length"], box["width"]]]
            }]
        })
    return annotations