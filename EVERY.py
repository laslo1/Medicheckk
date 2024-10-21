import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier('C:/Users/sould/OneDrive/Documents/GitHub/Medicheckk/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    return faces

def butter_bandpass_filter(data, order=4, lowcut=0.7, highcut=2.0, fs=30):
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
        b, g, r = cv2.split(roi)
        avg_intensity = np.mean(g)  # Use the average intensity of the green channel
        intensity_values.append(avg_intensity)
    
    # Apply bandpass filter
    if len(intensity_values) > 30:
        filtered_intensity_values = butter_bandpass_filter(intensity_values, order=4, lowcut=0.7, highcut=2.0, fs=fps)
    else:
        filtered_intensity_values = intensity_values
    
    # Find peaks in the filtered intensity values
    peaks, _ = find_peaks(filtered_intensity_values, distance=fps*0.450)
    
    # Extract peak times
    peak_times = np.array(peaks) / fps
    if len(peak_times) < 2:
        return 0, intensity_values, peaks
    
    intervals = np.diff(peak_times)
    
    # Calculate heart rate
    avg_interval = np.mean(intervals)
    if avg_interval == 0:
        return 0, intensity_values, peaks
    
    hr = 60 / avg_interval
    
    return hr, filtered_intensity_values, peaks

def calculate_spo2(frames, face, fps):
    (x, y, w, h) = face
    red_intensity = []
    green_intensity = []
    
    # Extract red and green channel intensity
    for frame in frames:
        roi = frame[y:y+h, x:x+w]
        b, g, r = cv2.split(roi)
        red_intensity.append(np.mean(r))
        green_intensity.append(np.mean(g))
    
    # Apply bandpass filter
    if len(red_intensity) > 30 and len(green_intensity) > 30:
        filtered_red = butter_bandpass_filter(red_intensity, order=4, lowcut=0.7, highcut=2.0, fs=fps)
        filtered_green = butter_bandpass_filter(green_intensity, order=4, lowcut=0.7, highcut=2.0, fs=fps)
    else:
        filtered_red = red_intensity
        filtered_green = green_intensity
    
    # Calculate AC and DC components for red and green channels
    ac_red = np.std(filtered_red)
    dc_red = np.mean(red_intensity)
    ac_green = np.std(filtered_green)
    dc_green = np.mean(green_intensity)
    
    if dc_red == 0 or dc_green == 0:
        return None  # Avoid division by zero
    
    # Ratio of Ratios (RoR)
    r = (ac_red / dc_red) / (ac_green / dc_green)
    
    # Estimate SpO2
    spo2 = 110 - 25 * r
    return spo2

def plot_hr(intensity_values, filtered_intensity_values, peaks):
    # Plot the original intensity in one window
    plt.figure()
    plt.plot(intensity_values, label="Original Intensity")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.title("Original Intensity")
    plt.legend()
    
    # Plot the filtered intensity and peaks in another window   
    plt.figure()
    plt.plot(filtered_intensity_values, label="Filtered Intensity", linestyle='--')
    plt.plot(peaks, [filtered_intensity_values[i] for i in peaks], "x", label="Peaks")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.title("Filtered Intensity with Peaks")
    plt.legend()
    
    # Show both figures
    plt.show()

def main():
    cap = cv2.VideoCapture(0)
    fps = 30  # 30 frames per second
    frames = []
    frame_count = 0
    max_frames = 210
    plt.ion()  # Enable interactive mode for real-time plotting
    
    # Capture all frames
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_face(frame)
        frames.append(frame)

        # Draw the blue rectangle around the detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue rectangle for the detected face
        
        # Display frame count on screen
        cv2.putText(frame, f"Frames: {frame_count}/{max_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        
        frame_count += 1
        key = cv2.waitKey(1)
        if key == ord('Q') or key == ord('q'):  # End if 'Q' or 'q' is pressed
            break
    
    # Process the captured frames
    if frames:
        for face in detect_face(frames[0]):
            hr, filtered_intensity_values, peaks = calculate_hr(frames, face, fps)
            spo2 = calculate_spo2(frames, face, fps)
            
            # Save results to a CSV file
            df = pd.DataFrame({"Heart rate": [hr], "SpO2": [spo2]})
            df.to_csv("data.csv", index=False)
            
            # Generate plots
            plot_hr([np.mean(cv2.split(f)[1]) for f in frames], filtered_intensity_values, peaks)
            
            print(f"Your heart rate is: {hr:.2f}")
            print(f"Your oxygen saturation (SpO2) is: {spo2:.2f}%")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()