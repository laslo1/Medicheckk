import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier('C:/Users/sould/OneDrive/Documents/GitHub/Contactless_rPPG_Python_Project/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    return faces

def calculate_hr(frames, face, fps):  # Pass 'frame' as an argument here
    (x, y, w, h) = face
    intensity_values = []

    for frame in frames:
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray)
        intensity_values.append(avg_intensity)

    peaks, _ = find_peaks(intensity_values, distance=fps/2)  # At least half a second between peaks

    # Calculate HR
    num_beats = len(peaks)
    duration_seconds = len(frames) / fps
    hr = (num_beats / duration_seconds) * 60
    return hr
    #Process for RR
def calculate_rr(frames, face, fps):  # Updated to accept 'frames' and 'fps'
    (x, y, w, h) = face
    intensity_values = []

    for frame in frames:
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #Convert the ROI to a gray scale
        avg_intensity = np.mean(gray) #The avarage gray of the photo
        intensity_values.append(avg_intensity) # the avarage gray of the photo will be included on the list

    # Use find_peaks to detect the breathing cycles
    peaks, _ = find_peaks(intensity_values, distance=fps * 2)  # Assume at least 3 seconds between breaths #//PROBLEMA EN ESTA LINEA COMO SE CALCULA
    #Select the peaks front all the frames 

    # Calculate respiratory rate (breaths per minute)
    num_breaths = len(peaks)
    duration_seconds = len(frames) / fps
    if num_breaths == 0:
        rr = 0
    else:
        rr = (num_breaths / duration_seconds) * 60  # Convert to breaths per minute

    return rr

def main():
    cap = cv2.VideoCapture(0)
    fps = 60  # Assuming 30 frames per second

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_face(frame)
        frames.append(frame)

        for face in faces:
            hr = calculate_hr(frames, face, fps)  # Pass 'frames', 'face', and 'fps' to the function
            rr = calculate_rr(frames, face, fps)  # Pass 'frames', 'face', and 'fps' to the function

            (x, y, w, h) = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a blue rectangle around the face

            # Display heart rate and respiratory rate on the frame
            cv2.putText(frame, f"Heart rate: {hr:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Respiratory rate: {rr:.2f}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Display a constant number on the frame
            constant_number = 42  # Change this to the desired constant number
            cv2.putText(frame, f"Constant: {constant_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            df = pd.DataFrame({"Heart rate": [hr], "Respiratory rate": [rr]})
            df.to_csv("data.csv", index=False)  # Avoid adding row numbers

        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1)
        if key == ord('Q') or key == ord('q'):  # Quit if 'Q' or 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
