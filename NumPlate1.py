import streamlit as st
import tempfile
import pytesseract
import numpy as np
import os

# Error handling for OpenCV import
try:
    import cv2
except ModuleNotFoundError as e:
    st.error(f"OpenCV (cv2) is not installed. Please install it using 'pip install opencv-python-headless'. Error details: {e}")
    st.stop()

# Error handling for Tesseract OCR import
try:
    import pytesseract
except ModuleNotFoundError as e:
    st.error(f"Pytesseract is not installed. Please install it using 'pip install pytesseract'. Error details: {e}")
    st.stop()

# Check if Tesseract OCR executable is available
if not pytesseract.pytesseract.tesseract_cmd or not os.path.isfile(pytesseract.pytesseract.tesseract_cmd):
    st.error("Tesseract OCR executable not found. Please ensure Tesseract is installed and 'tesseract_cmd' is correctly set.")
    st.stop()

# Define other necessary components for the Streamlit app
def extract_num(frame):
    global read
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nplate = cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in nplate:
            # Calculate the margin values for cropping
            a, b = int(0.02 * frame.shape[0]), int(0.025 * frame.shape[1])

            # Ensure the coordinates are within the image boundaries
            y1, y2 = max(0, y + a), min(frame.shape[0], y + h - a)
            x1, x2 = max(0, x + b), min(frame.shape[1], x + w - b)

            # Check if the coordinates are valid
            if y1 >= y2 or x1 >= x2:
                st.error(f"Error: Invalid coordinates for cropping: ({x1}, {y1}), ({x2}, {y2})")
                continue

            plate = frame[y1:y2, x1:x2, :]

            if plate.size == 0:
                st.error("Error: Extracted plate region is empty.")
                continue

            # Image processing
            kernel = np.ones((1, 1), np.uint8)
            plate = cv2.dilate(plate, kernel, iterations=1)
            plate = cv2.erode(plate, kernel, iterations=1)
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            _, plate = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

            read = pytesseract.image_to_string(plate)
            read = ''.join(e for e in read if e.isalnum())
            stat = read[0:2]

            if read:
                st.write(f"Detected number plate: {read}")

            # Your license plate processing code here...

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

# Your Streamlit app UI and main function
def main():
    st.title("Vehicle Number Plate Recognition")

    # Example of loading an image
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        extract_num(frame)

if __name__ == "__main__":
    main()
