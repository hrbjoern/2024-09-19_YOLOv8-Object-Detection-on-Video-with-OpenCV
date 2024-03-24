# YOLO Object Detection on Video

This project implements YOLOv8 (You Only Look Once) object detection on a video using Python and OpenCV. YOLO is a state-of-the-art, real-time object detection system that achieves high accuracy and fast processing times.

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/your_username/yolo-object-detection.git
    ```

2. Install the required libraries:

    ```
    pip install -r requirements.txt
    ```

3. Download the YOLO weights file (e.g., `yolov8l.pt`) and place it in the `Yolo-Weights` directory.

## Usage

1. Replace `bikes.mp4` in the `Videos` directory with your desired video file.

2. Run the Python script:

    ```
    python yolo_detection.py
    ```

3. The script will perform object detection on the video frames using YOLO and save the output frames with bounding boxes in the `output_frames` directory.

4. Once the processing is complete, the script will create a new video (`output_video.mp4`) with the object detection results.

## Output Video

Here is the output video with object detection applied:

<video width="640" height="480" controls>
  <source src="output_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Customization

- You can customize the classes to be detected by modifying the `classNames` list in the Python script.
- Adjust the confidence threshold and other parameters for object detection according to your requirements.

## Credits

- [YOLOv8](https://github.com/ultralytics) by Ultralytics for the YOLO implementation.
- [OpenCV](https://opencv.org/) for image and video processing.
- [cvzone](https://github.com/cvzone/cvzone) for drawing bounding boxes and text on images.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README template provides a clear overview of the project, installation instructions, usage guidelines, customization options, credits, and licensing information. Customize it according to your project's specific requirements and preferences.

Feel free to reach out with any questions or suggestions!
