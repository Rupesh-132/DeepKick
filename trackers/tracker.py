
from ultralytics import YOLO
import supervision as sv
import pickle
import os 
import sys
sys.path.append("../")
from utils import get_bbox_width_height, get_center_of_bbox
import cv2
import numpy as np

class Tracker:
    def __init__(self,model_path):
        
        self.model  = YOLO(model_path)
        self.tracker = sv.ByteTrack()   ## to keep the track of the detected ojects from one frame to the another
    
    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections += detections_batch
            
        return  detections
        
        
    def get_object_tracks(self,frames,read_from_stub = False, stub_path = None):
        
        if read_from_stub  and stub_path is not None and os.path.exists(stub_path):
                with open(stub_path,"rb") as f:
                    tracks = pickle.load(f)
                
                return tracks
        
        # get the detections first and then tracking
        detections = self.detect_frames(frames)
        
        # overwriting the goalkeepers as players in the dtected frames
        
        tracks =  {
            "players":[],
            "referees": [],
            "ball":[]
        }
        for frame_num,detection in enumerate(detections):
            
            # access the detected class from the frames
            class_names = detection.names    # names: {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
            class_names_inv = {v: k for k,v in class_names.items()} 
            
            print(class_names)
            # convert the detections to supervision detections
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Convert the goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                
                if class_names[class_id] == "goalkeeper":
                    # swapping the id of the goalkeer with the player id
                    detection_supervision.class_id[object_ind] = class_names_inv["player"]
            
            # Track objects
            # this is going to add the tracker objects to the detections
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)      
            
            tracks["players"].append({}) # key will the be tracker id and the value would be the bounding box
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                tracker_id = frame_detection[4]
                
                if class_id == class_names_inv["player"]:
                    tracks["players"][frame_num][tracker_id] = {"bbox":bbox}
                    
                if class_id == class_names_inv["referee"]:
                    tracks["referees"][frame_num][tracker_id] = {"bbox":bbox}
                
                
            # Since the ball per frame is going to be same in all the frames we will loop through the detections without tracks
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                tracker_id = frame_detection[4]
                
                if class_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}  # since there is 1 ball id hardcoding the tracker_id for the ball
            
            # print(detections_with_tracks)
            
            # saving the tracks to avoid reading detections and waiting for the output again and again
            if stub_path is not None:
                with open(stub_path,"wb") as f:
                    pickle.dump(tracks,f)
                    
            return tracks
        
    def draw_ellipse(self,frame,bbox,color,tracker_id=None):
        y2 = int(bbox[3])
        
        X_center, _ = get_center_of_bbox(bbox)
        width, _ = get_bbox_width_height(bbox)
        cv2.ellipse(frame, 
                    center =(X_center, y2),
                   axes = (int(width), int(0.35 * width)),
                   angle = 0.0,
                   startAngle = -45,
                   endAngle = 235,
                   color = color,
                   thickness = 2,
                   lineType = cv2.LINE_4)
        
        # Draw the tracker ID along with the each player
        rectangle_width  =  40
        rectange__height = 20
        x1_rectangle = int(X_center - rectangle_width//2)
        x2_rectangle = int(X_center + rectangle_width//2)
        y1_rectangle = int(y2 - rectange__height//2) +  15
        y2_rectangle = int(y2 + rectange__height//2) + 15
        
        if tracker_id is not None:
            cv2.rectangle(frame, (x1_rectangle, y1_rectangle), (x2_rectangle, y2_rectangle), color, cv2.FILLED)
            
            x1_text = x1_rectangle + 12
            if tracker_id > 99:
                x1_text -= 10
        
            cv2.putText(frame, str(tracker_id), (int(x1_text), int(y1_rectangle + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        return frame
    
    def draw_triange_for_ball(self,frame,bbox,color):
        
        y = int(bbox[1]) # we want the triangle to be drawn on the top of the ball
        X_center, _ = get_center_of_bbox(bbox)
        
        triangle_points = np.array(
            [
                [X_center,y],
                [X_center - 10,y - 20],
                [X_center + 10,y - 20]
            ]
        )
        cv2.drawContours(frame, [triangle_points], 0, color, thickness=cv2.FILLED) # this will fill the triangle with the color
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        
        return frame
        
        
    def draw_annotations(self, video_frames, tracks):
        
        output_video_frames  = []
        
        for frame_num, frame in enumerate(video_frames):
            
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num] 
            ball_dict = tracks["ball"][frame_num]
            
            # Draw players
            for tracker_id,player in player_dict.items():
                color = player.get("team_color",  (0, 0, 255)) # default color is red
                # Draw the bounding box around the player
                bbox = player["bbox"]
                frame = self.draw_ellipse(frame, bbox, color, tracker_id)
                
            # Draw referees
            for _,referee in referee_dict.items():
                # Draw the bounding box around the referee
                bbox = referee["bbox"]
                frame = self.draw_ellipse(frame, bbox, (0, 255, 255))
           
            # Draw ball
            for tracker_id,ball in ball_dict.items():
                # Draw the bounding box around the ball
                bbox = ball["bbox"]
                frame = self.draw_triange_for_ball(frame, bbox, (0, 255, 0))
                
            output_video_frames.append(frame)
            
        return output_video_frames
                
                
                
               
            
            