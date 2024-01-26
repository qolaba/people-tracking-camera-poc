import logging
import uuid
import threading
import cv2
from pytube import YouTube
from ultralytics import YOLO
import os

# Configure logging to display time, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_youtube_video(url, output_path='.'):
    """
    Downloads a YouTube video to a specified path.

    Args:
        url (str): The URL of the YouTube video.
        output_path (str): The directory path to save the downloaded video.

    Returns:
        str: The filename of the downloaded video.
    """
    try:
        yt = YouTube(url)  # Create a YouTube object for the given URL
        video_stream = yt.streams.get_highest_resolution()  # Get the highest resolution video stream
        unique_filename = str(uuid.uuid4())[:8]  # Generate a unique filename
        video_stream.download(output_path, filename=f"{unique_filename}.mp4")  # Download the video
        return f"{unique_filename}.mp4"  # Return the filename of the downloaded video
    except Exception as e:
        logging.error(f"An error occurred while downloading the video: {e}")  # Log any errors during download
        return None

def run_tracker_in_thread(filename, model, file_index):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.
    """
    try:
        video = cv2.VideoCapture(filename)  # Capture video from the given filename

        while True:
            ret, frame = video.read()  # Read frames from the video
            if not ret:
                break  # Exit the loop if no more frames available

            results = model.track(frame, persist=True)  # Track objects in the frame using the YOLO model
            res_plotted = results[0].plot()  # Plot the tracking results on the frame
            cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)  # Display the plotted frame

            if cv2.waitKey(1) == ord('q'):  # Exit if 'q' key is pressed
                break

        video.release()  # Release the video capture object
    except Exception as e:
        logging.error(f"Error in tracking thread {file_index}: {e}")  # Log any errors during tracking

def main():
    video_url = "https://www.youtube.com/watch?v=7V6hq5dM5oI"  # URL of the YouTube video to download
    file_name = download_youtube_video(video_url)  # Download the video
    # file_name = "people.mp4"  # Download the video
    if file_name:
        model = YOLO('yolov8n.pt')  # Load the YOLOv8 model
        model2 = YOLO('yolov8n-seg.pt')  # Load another YOLOv8 model variant
        video_file1 = file_name  # Use the downloaded video for tracking
        video_file2 = 0  # Use the default webcam for tracking

        # Create and start threads for tracking
        tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model, 1), daemon=True)
        tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, 2), daemon=True)
        tracker_thread1.start()
        tracker_thread2.start()

        # Wait for the tracking threads to finish
        tracker_thread1.join()
        tracker_thread2.join()

        cv2.destroyAllWindows()  # Close all OpenCV windows
        os.remove(file_name)

if __name__ == "__main__":
    main()
