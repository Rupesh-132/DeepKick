# ⚽️ DeepKick(Football Analysis System)

This project presents a comprehensive Computer Vision and Machine Learning pipeline for **analyzing player movements and gameplay patterns** in football match videos.  
It integrates multiple advanced techniques to deliver deep tactical insights, player tracking, ball possession analysis, and real-world position mapping.

---

## 🚀 Key Features

### 🧐 1. Comprehensive Object Detection and Tracking
- Detect players, goalkeepers, referees, and footballs using a **custom YOLO model**.
- Track entities across frames using **ByteTracker**, maintaining consistent IDs for players and objects.

### 🟘️ 2. Field Keypoint Detection
- Detect keypoints on the football field using **YOLO Pose estimation**.
- Apply **histogram equalization** and **contrast enhancement** to improve keypoint detection accuracy.
- Map key field areas to enable correct spatial understanding for subsequent analysis.

### 🎽 3. Player Club Assignment
- Classify players into clubs based on **jersey color extraction**.
- Use **K-Means clustering** to isolate dominant jersey colors and **match players to teams**.
- Differentiate between player jerseys and goalkeeper jerseys effectively.

### 📍 4. Real-World Position Mapping
- Perform **Perspective Transformation** to map player positions from the broadcast view to a top-down view of the pitch.
- Compute a **homography matrix** using detected keypoints for accurate positional analysis.

### 🧹 5. Dynamic Voronoi Diagram
- Generate dynamic **Voronoi diagrams** around players based on real-world positions.
- Visualize each player’s zone of control and spatial dominance on the field in real-time.

### ⚽️ 6. Ball Possession Calculation
- Assign ball possession to the nearest player using **Euclidean distance**.
- Implement validation checks to ignore invalid detections.
- Track possession periods, including **grace periods** to handle ball occlusions or fast movements.

### 🚀 7. Speed Estimation
- Estimate player speeds by tracking movement between frames and applying **pixel-to-meter scaling**.
- Smooth speed measurements using a **moving average**.
- Apply realistic speed caps (e.g., 40 km/h) to maintain data integrity.

### 🎥 8. Live Video Preview
- Enable real-time visualization of tracking, mapping, and player annotations during video processing.

### 📂 9. Tracking Data Storage
- Save all tracking outputs (positions, identities, speed, possession) in structured **JSON files** for post-match analysis and machine learning tasks.

---

## 🏆 Technologies Used
- Python
- OpenCV
- YOLOv8 / YOLO Pose
- Scikit-learn (K-Means)
- NumPy
- ByteTrack (Object Tracking)

---

## 📊 Applications
- Player and team tactical analysis
- Player speed and movement profiling
- Ball possession trends over time
- Automated video tagging and event detection
- Support tools for scouting, coaching, and match analysis

> 🌟 **Advanced Computer Vision system for Football Player Analysis, Tactical Insights, and Game Understanding.**
