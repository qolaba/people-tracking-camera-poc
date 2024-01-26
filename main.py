from pytube import YouTube
from ultralytics import YOLO
import uuid

def download_youtube_video(url, output_path='.'):
    try:
        # Create a YouTube object
        yt = YouTube(url)

        # Get the highest resolution stream
        video_stream = yt.streams.get_highest_resolution()

        # Download the video to the specified output path
        unique_filename = str(uuid.uuid4())[:8]

        video_stream.download(output_path, filename= f"{unique_filename}.mp4")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return f"{unique_filename}.mp4"
    

# Example usage
video_url = "https://www.youtube.com/watch?v=WvhYuDvH17I"
file_name = download_youtube_video(video_url)

model = YOLO('yolov8n.pt')

results = model.track(source=file_name, tracker="bytetrack.yaml", save=True) 
