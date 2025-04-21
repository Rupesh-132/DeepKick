
def get_center_of_bbox(bbox):
  
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    return center_x, center_y


def get_bbox_width_height(bbox):
   
    x1, y1, x2, y2 = bbox
    width = int(x2 - x1)
    height = int(y2 - y1)
    
    return width, height