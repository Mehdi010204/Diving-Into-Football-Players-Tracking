from Utils import read_video, save_video
from Tracker import Tracker
from Teams import TeamAssigner

def main():
    # Reading video
    video_frames = read_video('Tests\Video\Other_input_video.mp4')

    # Initializing tracker
    tracker = Tracker('Finetuned Yolo/best(1).pt')

    tracks = tracker.get_object_tracks(video_frames)
    
    # Assigning teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # Looping over each player in each frame and assigning them to colour team
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Drawing annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Saving video
    save_video(output_video_frames, 'Output_videos/other_output_video.avi')



if __name__ == '__main__':
    main()