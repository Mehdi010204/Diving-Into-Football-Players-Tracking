from ultralytics import YOLO
import os
import supervision as sv
import pickle
import cv2
import numpy as np
from Utils import get_center_of_bbox, get_bbox_width


class Tracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

        self.tracker = sv.ByteTrack()


    def detect_frames(self, frames):

        batch_size = 50  
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections
    

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            # mapping class and names{0:person, 1: goal,.. etc}
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Converting to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision)

            
            # Appending the track dictionnary with bounding boxes for each frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]  
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # If a stub_path was provided, save the tracking data to this file using pickle
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    

    def draw_ellipse(self, frame, bbox, color, track_id=None):

        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Drawing ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Drawing rectangle
        # Defining the dimensions of rectangle to be drawn
        rectangle_width = 40
        rectangle_height = 20
        
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:  # checking if track_id was provided then

            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        return frame


    def draw_triangle(self, frame, bbox, color):
        # y is set to the integer value of the second element of bbox
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Defining the points of the triangle
        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        # Drawing filled triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)

        # Drawing triangle outline
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame
    

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        # Here we starting a loop that iterates through each frame in video_frames. enumerate() is used to get both the index (frame_num) and the frame itself.
        for frame_num, frame in enumerate(video_frames):
            # creating a copy of the current frame to avoid modifying the original frame.
            frame = video_frames[frame_num]

            # Retrieving dictionaries containing tracking information for players, the ball, and referees for the current frame.
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Drawing Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(
                    frame, player["bbox"], color, track_id)

                # if player doesn't have ball then we can draw just a triangle with colour red
                if player.get('has_ball', False):
                    frame = self.draw_triangle(
                        frame, player["bbox"], (0, 0, 255))

            # Drawing Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(
                    frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames