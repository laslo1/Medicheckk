import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier('C:/Users/sould/OneDrive/Documents/GitHub/Medicheckk/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    return faces

def butter_bandpass_filter(data, order=4, lowcut=0.5, highcut=2.0, fs=30):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def calculate_hr_and_spo2(frames, face, fps):
    (x, y, w, h) = face
    intensity_values = []
    red_intensity = []
    green_intensity = []
    
    for frame in frames:
        roi = frame[y:y+h, x:x+w]
        _, g, r = cv2.split(roi)
        avg_green_intensity = np.mean(g)
        avg_red_intensity = np.mean(r)
        
        intensity_values.append(avg_green_intensity)
        red_intensity.append(avg_red_intensity)
        green_intensity.append(avg_green_intensity)
    
    filtered_intensity_values = butter_bandpass_filter(intensity_values, lowcut=0.7, highcut=2.0, fs=fps)
    peaks, _ = find_peaks(filtered_intensity_values, distance=fps*0.450)
    
    peak_times = np.array(peaks) / fps
    intervals = np.diff(peak_times)
    hr = 60 / np.mean(intervals) if len(intervals) > 0 else 0
    
    filtered_red = butter_bandpass_filter(red_intensity, lowcut=0.5, highcut=2.0, fs=fps)
    filtered_green = butter_bandpass_filter(green_intensity, lowcut=0.5, highcut=2.0, fs=fps)
    
    ac_red = np.std(filtered_red)
    dc_red = np.mean(red_intensity)
    ac_green = np.std(filtered_green)
    dc_green = np.mean(green_intensity)
    
    r = (ac_red / dc_red) / (ac_green / dc_green) if dc_red != 0 and dc_green != 0 else 0
    spo2 = 110 - 25 * r if r != 0 else 0
    
    return hr, spo2, filtered_intensity_values, peaks, filtered_red, filtered_green

def plot_hr_and_spo2(filtered_hr, peaks, filtered_red, filtered_green):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(filtered_hr, label="Filtered HR Signal", color='blue')
    plt.plot(peaks, [filtered_hr[i] for i in peaks], "x", color='red', label="Peaks")
    plt.title("Heart Rate Signal with Peaks")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(filtered_red, label="Filtered Red Intensity", color='orange')
    plt.plot(filtered_green, label="Filtered Green Intensity", color='green')
    plt.title("Red and Green Intensities for SpO2")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    cap = cv2.VideoCapture(0)
    fps = 30
    frames = []
    frame_count = 0
    max_frames = 210
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_face(frame)
        frames.append(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.putText(frame, f"Frames: {frame_count}/{max_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    if frames:
        faces = detect_face(frames[0])
        if len(faces) > 0:
            hr, spo2, filtered_hr, peaks, filtered_red, filtered_green = calculate_hr_and_spo2(frames, faces[0], fps)
            print(f"Detected HR: {hr} bpm, SpO2: {spo2}%")
            plot_hr_and_spo2(filtered_hr, peaks, filtered_red, filtered_green)
        else:
            print("No face detected for analysis.")
    else:
        print("No frames captured.")

if __name__ == "__main__":
    main()
