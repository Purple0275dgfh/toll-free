import streamlit as st
import cv2
import pytesseract
import numpy as np

# Initialize Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Streamlit app
def main():
    st.title("Number Plate Recognition")

    # Upload video file
    uploaded_file = st.file_uploader("Choose a video...", type="mp4")

    if uploaded_file is not None:
        # Read the video file
        video = cv2.VideoCapture(uploaded_file.name)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
            plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in plates:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                plate = frame[y:y + h, x:x + w]
                text = pytesseract.image_to_string(plate)
                st.write(f"Detected Plate: {text}")

            # Display frame
            st.image(frame, channels="BGR")

        video.release()


if __name__ == "__main__":
    main()
