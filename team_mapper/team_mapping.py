import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamMapper:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}  # Dictionary to store player id and their team player_id : 1 or 2
    
    def get_clustering_model(self, image):
        # Reshape the imahe into 2D array
        image_2d = image.reshape((-1, 3))

        # performs k-means clustering
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1).fit(image_2d)
        
        return kmeans
        
    
    def get_player_color(self, frame, bbox):
        
        # Extract the region of interest (ROI) from the frame using the bounding box coordinates i.e crop the players from the frame
        x1, y1, x2, y2 = bbox
        roi_image = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # get the top half of the player to get the jersey color
        top_half_image = roi_image[0:int(roi_image.shape[0] / 2), :]
        
        # get the clustering model
        kmeans = self.get_clustering_model(top_half_image)
        
        # get the cluster labels and cluster centers
        cluster_labels = kmeans.labels_ 
        
        # resape the labels to the original image shape
        clustered_image = cluster_labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        
        # get the player cluster
        corner_labels = clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]

        non_player_label = np.bincount(corner_labels).argmax()
        player_cluster_label = 1 - non_player_label
        
        # get the player_cluster color
        player_cluster_color = kmeans.cluster_centers_[player_cluster_label]
        
        return player_cluster_color
        

    def map_team_colors(self, frame, player_detections):
        
        
        player_colors  = [] # put the color of all the players detected in the frame in a list
        
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            
            player_colors.append(player_color)
        
        kmeans  = KMeans(n_clusters=2, init="k-means++",n_init=1).fit(player_colors)
        cluster_labels = kmeans.labels_
        
        self.kmeans = kmeans
        
        # get the color for each team
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
    
    
    def get_player_teams(self,frame,player_bbox,player_id):
        
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict([player_color])[0] + 1
        
        self.player_team_dict[player_id] = team_id
        
        return team_id