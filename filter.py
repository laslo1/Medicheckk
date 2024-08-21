import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier('C:/Users/sould/OneDrive/Documents/GitHub/Medicheckk/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    return faces
def bandpass_filter(data, lowcut, highcut, fs=30, order=2):  # Reduced order to 2
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

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

    intensity_values = np.array(intensity_values)
    #intensity_values = (intensity_values - np.mean(intensity_values)) / np.std(intensity_values)
    
    # Check if the signal length is sufficient for filtering
    #if len(intensity_values) > 20:  # 30 is an arbitrary threshold, can adjust
   #     filtered_signal = bandpass_filter(intensity_values, 0.5, 3.0, fps)
    #else:
   #     filtered_signal = intensity_values  # If too short, skip filtering

    # Detect peaks in the filtered signal
    peaks, _ = find_peaks(intensity_values, distance=fps/2)  
        # Calculate HR
    num_beats = len(peaks) #  la cantidad de picos hallados
    duration_seconds = len(frames) / fps # cuantos frames hay entre fps <<<< Probablemente el error esta aqui len(frames) todos los frames tomados
    #intentar probando con el valor 6 ^^^^^
    #hr=(num_beats/6)*60
    hr = (num_beats / duration_seconds) * 60 # 10 picos en 180 frames pero se toman mas de 180 ☻ (Por eso el posible error)
    #Se supone que la ecuacion es #DePicos*10 en 6 segundos
    return hr, intensity_values, peaks 
#is better to work with the  number of beats that the len
def calculate_rr(frames, face, fps):
    (x, y, w, h) = face
    intensity_values = []
    for frame in frames:
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray)
        intensity_values.append(avg_intensity)
    peaks, _ = find_peaks(intensity_values, distance=fps * 2, height=0.1)
    num_breaths = len(peaks)
    duration_seconds = len(frames) / fps
    if num_breaths == 0:
        rr = 0
    else:
        rr = (num_breaths / duration_seconds) * 60  # Convert to breaths per minute
    return rr

def plot_hr(intensity_values, peaks):
    plt.clf()
    plt.plot(intensity_values, label="Intensity")
    plt.plot(peaks, [intensity_values[i] for i in peaks], "x", label="Peaks")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.title("Heart Rate Peaks")
    plt.legend()

def main():
    cap = cv2.VideoCapture(0)
    fps = 30
    frames = []
    frame_count = 0
    max_frames = 210
    plt.ion()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count < max_frames:
            faces = detect_face(frame)
            frames.append(frame)
            for face in faces:
                hr, intensity_values, peaks = calculate_hr(frames, face, fps)
                rr = calculate_rr(frames, face, fps)
                (x, y, w, h) = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Heart rate: {hr:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Respiratory rate: {rr:.2f}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                constant_number = 0
                cv2.putText(frame, f"Constant: {constant_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                df = pd.DataFrame({"Heart rate": [hr], "Respiratory rate": [rr]})
                df.to_csv("data.csv", index=False)
                plot_hr(intensity_values, peaks)
                plt.pause(0.01)
                cv2.imshow('frame', frame)
        else:
            cv2.putText(frame, str(hr) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) #The text was change for the value of hr
        cv2.imshow('frame', frame)
        frame_count += 1
        key = cv2.waitKey(1)
        if key == ord('Q') or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
