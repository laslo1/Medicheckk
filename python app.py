import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier('C:/Users/sould/OneDrive/Documents/GitHub/Medicheckk/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    return faces

def butter_bandpass_filter(data, order=2, lowcut=0.7, highcut=1.0, fs=30):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def calculate_hr(frames, face, fps):  
    (x, y, w, h) = face
    intensity_values = []
    for frame in frames:
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray)
        if len(intensity_values) >= 180:
            intensity_values.pop(0)  
        intensity_values.append(avg_intensity)
        
    # Check if there is enough data to filter
    if len(intensity_values) > 170:
        # Apply Butterworth bandpass filter
        filtered_intensity_values = butter_bandpass_filter(intensity_values, order=2, lowcut=0.7, highcut=2.5, fs=fps)
    else:
        # If not enough data, use raw intensity values
        filtered_intensity_values = intensity_values
    
    # Find peaks
    peaks, _ = find_peaks(filtered_intensity_values, distance=fps/2)  
    
    # Calculate intervals between peaks
    peak_times = np.array(peaks) / fps
    if len(peak_times) < 2:
        return 0, intensity_values, filtered_intensity_values, peaks   
    
    intervals = np.diff(peak_times)
    
    # Calculate HR in beats per minute
    avg_interval = np.mean(intervals)  # Average interval between peaks in seconds
    if avg_interval == 0:
        return 0, intensity_values, filtered_intensity_values, peaks
    
    hr = 60 / avg_interval
    
    return hr, intensity_values, filtered_intensity_values, peaks

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

def plot_hr(intensity_values, filtered_intensity_values, peaks):
    plt.figure("Intensity Plot")
    plt.subplot(2, 1, 1)
    plt.cla()  # Clear the current axes
    plt.plot(intensity_values, label="Original Intensity")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.title("Original Intensity")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.cla()  # Clear the current axes
    plt.plot(filtered_intensity_values, label="Filtered Intensity", linestyle='--')
    plt.plot(peaks, [filtered_intensity_values[i] for i in peaks], "x", label="Peaks")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.title("Filtered Intensity")
    plt.legend()

    plt.tight_layout()
    plt.draw()  # Update the figure
    plt.pause(0.01)  # Pause to allow the plot to be updated

def main():
    cap = cv2.VideoCapture(0)
    fps = 30  # 30 frames per second
    frames = []
    frame_count = 0
    max_frames = 200
    
    # Set up initial plots
    plt.ion()  # Enable interactive mode
    plot_hr([], [], [])  # Initialize plot with empty data

    while True:
        ret, frame = cap.read()
        if not ret:
            break   
        if frame_count < max_frames:
            faces = detect_face(frame)
            frames.append(frame)
            for face in faces:
                hr, intensity_values, filtered_intensity_values, peaks = calculate_hr(frames, face, fps)
                rr = calculate_rr(frames, face, fps)
                (x, y, w, h) = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Heart rate: {hr:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Respiratory rate: {rr:.2f}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                constant_number = 0  # Change this to the desired constant number
                cv2.putText(frame, f"Constant: {constant_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                df = pd.DataFrame({"Heart rate": [hr], "Respiratory rate": [rr]})
                df.to_csv("data.csv", index=False)
                plot_hr(intensity_values, filtered_intensity_values, peaks)  # Update the plot with new data
                cv2.imshow('frame', frame)
        else:
            cv2.putText(frame, str(hr), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Stop recording
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
