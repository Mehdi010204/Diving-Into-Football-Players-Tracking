# Diving-Into-Football-Players-Tracking

This project aims to track football players, referees, and the ball using computer vision techniques. It utilizes a fine-tuned YOLOv8 model to classify four distinct classes in football match footage: players, goalkeepers, referees, and ball. This model allows for accurate object detection and is combined with tracking algorithms to display the movement of the objects throughout the video.\\

## Overview
- **Object Detection**: Fine-tuned YOLOv8 on a custom dataset containing 4 classes (ball, players, goalkeepers, referees) to improve detection accuracy and avoid generalized classification.
  
- **Tracking**: 
  - Used ellipses to represent players and triangles to represent the ball.
  - KMeans clustering was employed to differentiate between the two teams based on shirt colors.
  
- **Key Features**:
  - **Player Tracking**: Players are tracked with ellipses, and their shirt colors (from the clustering) are used for identification.
  - **Ball Tracking**: The ball is tracked with triangles, though further work is required to improve accuracy between frames.

## Technologies Used
- **YOLOv8**: Object detection model for identifying players, referees, ball, and goalkeepers.

- **OpenCV**: For video manipulation and drawing shapes for tracking.

- **KMeans**: Clustering algorithm to separate teams based on shirt colors.

- **NumPy & SciPy & Matplotlib**: For data processing, analysis and visualization.

## Future Improvements
- **Ball Tracking Enhancements**: The ball tracking can sometimes fail, especially in fast-moving scenarios or between frames. Improving its reliability is a future goal.
  
- **Players IDs**: One of the project's future goals is to assign fixed ID numbers to the players, letting them for example correspond to the shirt numbers.

- **Goalkeeper Identification**: Currently, goalkeepers are represented similarly to players. Plans include a more specific and distinctive display for goalkeepers.
  
- **Alternative Classification Algorithms**: While KMeans clustering is used to separate the teams, other classification algorithms will be explored to better differentiate between players on the field.
  
- **Radar Map Visualization**: The project will eventually include generating radar maps based on the output video. This will involve detecting key points in the field to analyze team possession, player movements, and create heatmaps for further tactical analysis, which will require fine-tuning YOLOv8 on a more specific dataset or using another model for key points detection.
