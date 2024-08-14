import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def detect_face(frame):
    face_cascade = cv2.CascadeClassifier('C:/Users/sould/OneDrive/Documents/GitHub/Medicheckk/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    return faces
def calculate_hr(frames, face, fps):  # Pass 'frame' as an argument here
    (x, y, w, h) = face
    intensity_values =  []
    for frame in frames:
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray)
        if len(intensity_values) >= 180:
             intensity_values.pop(0)  # Remove the oldest element if the list already has 180 elements
        intensity_values.append(avg_intensity)
    peaks, _ = find_peaks(intensity_values, distance=fps/2)  # At least half a second between peaks
    # Calculate HR
    num_beats = len(peaks)
    duration_seconds = len(frames) / fps
    hr = (num_beats / duration_seconds) * 60
    return hr, intensity_values, peaks
def calculate_rr(frames, face, fps):
    (x, y, w, h) = face
    intensity_values = []
    for frame in frames:
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray)
        intensity_values.append(avg_intensity)
    peaks, _ = find_peaks(intensity_values, distance=fps * 2)
    num_breaths = len(peaks)
    duration_seconds = len(frames) / fps
    if num_breaths == 0:
        rr = 0
    else:
        rr = (num_breaths / duration_seconds) * 60  # Convert to breaths per minute
    return rr
def plot_hr(intensity_values, peaks):
    plt.clf()  # Clear the previous plot
    plt.plot(intensity_values, label="Intensity")
    plt.plot(peaks, [intensity_values[i] for i in peaks], "x", label="Peaks")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.title("Heart Rate Peaks")
    plt.legend()
def main():
    cap = cv2.VideoCapture(0)
    fps = 30  # 30 frames per second
    frames = []
    frame_count= 0
    max_frames= 200
    plt.ion()  # Enable interactive mode for real-time plotting
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count< max_frames:
            faces = detect_face(frame)
            frames.append(frame)
            for face in faces:
                hr, intensity_values, peaks = calculate_hr(frames, face, fps)
                rr = calculate_rr(frames, face, fps)
                (x, y, w, h) = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Heart rate: {hr:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Respiratory rate: {rr:.2f}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                constant_number = 0  # Change this to the desired constant number
                cv2.putText(frame, f"Constant: {constant_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                df = pd.DataFrame({"Heart rate": [hr], "Respiratory rate": [rr]})
                df.to_csv("data.csv", index=False)
                plot_hr(intensity_values, peaks)  # Update the plot with new data
                plt.pause(0.01)  # Pause to update the plot
                cv2.imshow('frame', frame)
        else :
            cv2.putText(frame, "Data collection stopped", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) #Stop recording
        cv2.imshow('frame', frame)
        frame_count += 1
        key = cv2.waitKey(1)
        if key == ord('Q') or key == ord('q'):  # Quit if 'Q' or 'q' is pressed
            break
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Disable interactive mode
    plt.show()  # Show the final plot

if __name__ == "__main__":
    main()

