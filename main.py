from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_mapper import TeamMapper

def main():
   ## Read Video
   video_frames = read_video(r"data\input\08fd33_4.mp4")
   
   # Initialize the tracker
   tracker = Tracker(r"model\trained\weights\best.pt")
   
   tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path= r"stubs\tracks_stub.pkl")
   
   # Save the cropped image of the player
   # for tracker_id, player in tracks["players"][0].items():
   #    bbox = player["bbox"]
   #    frame = video_frames[0]
      
   #    # crop bbox from the frame
   #    cropped_image = frame[int(bbox[1]) :int(bbox[3]), int(bbox[0]):int(bbox[2])]
      
   #    # save the cropped image
   #    cv2.imwrite(f"data\output\player_cropped_img.jpg", cropped_image)
      
   #    break
   
   # Assign player team
   team_mapper = TeamMapper()
   team_mapper.map_team_colors(video_frames[0], tracks["players"][0])
   
   # Now we want to traverse each players in each frame and assign them a team
   for frame_num, player_track in enumerate(tracks["players"]):
      for player_id, track in player_track.items():
         team = team_mapper.get_player_teams(video_frames[frame_num], track["bbox"], player_id)
         # Add the team to the track   
         tracks["players"][frame_num][player_id]["team"] = team
         tracks["players"][frame_num][player_id]["team_color"] = team_mapper.team_colors[team]
         
         
   # Draw object annotations on the video frames
   # Draw object tracks
   output_video_frames = tracker.draw_annotations(video_frames, tracks)
   # Save the video
   save_video(output_video_frames,"data\output\output.avi")
    

if __name__ == "__main__":
    main()