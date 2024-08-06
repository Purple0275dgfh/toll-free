import streamlit as st
import cv2
import pytesseract
import numpy as np
import tempfile
from Database import Database
import re

# Set the correct path for Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize other configurations
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
license_plate_pattern = re.compile(r'^[A-Z]-\d{3}-[A-Z]{2}$')
db = Database()

Tolls = {
    "Car": 45,
    "Jeep": 45,
    "Van": 45,
    "LCV": 75,
    "Bike": 0,
    "Bus": 150,
    "Truck": 150
}

states = {
    "AN": "Andaman and Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam", "BR": "Bihar",
    "CG": "Chattisgarh", "CH": "Chandigarh", "DD": "Dadra and Nagar Haveli and Daman and Diu", "DL": "Delhi",
    "GA": "Goa", "GJ": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana", "JH": "Jharkhand",
    "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala", "KS": "Kyasamballi", "LA": "Ladakh",
    "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya", "MN": "Manipur", "MP": "Madhya Pradesh",
    "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan",
    "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana", "UK": "Uttarakhand",
    "UP": "Uttar Pradesh", "WB": "West Bengal", "R": "Romania", "N": "Norway", "L": "Luxembourg", "H": "Hungary",
    "K": "Kenya"
}


def extract_num(frame):
    global read
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
            st.error(f"Invalid coordinates for cropping: ({x1}, {y1}), ({x2}, {y2})")
            continue

        plate = frame[y1:y2, x1:x2, :]

        if plate.size == 0:
            st.error("Extracted plate region is empty.")
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

        if license_plate_pattern.match(read):
            if read not in visited:
                if "-" in stat:
                    stat = stat.replace("-", "")
                try:
                    ret = db.display_vehicle_details(read[0:-1])
                    if ret:
                        st.write('Vehicle belongs to', states[stat])
                        st.write("Plate number: ", read)
                        vehicle_type = Tolls[db.get_type(read[0:-1])]
                        st.write(f"Toll calculated: {vehicle_type}")
                    visited.add(read)
                except:
                    st.error('State not recognized!')

        cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (51, 51, 255), -1)

        plate_resized = cv2.resize(plate, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        st.image(plate_resized, caption="Detected Plate")
        break  # Break after processing the first frame

    return frame


def main():
    st.title("License Plate Recognition and Toll Calculation")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        ret = True
        visited = set()

        stop_button_pressed = False

        while ret:
            ret, frame = cap.read()
            if not ret:
                break

            frame = extract_num(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB")

            if st.button('Stop', key='stop_button'):
                stop_button_pressed = True
                break

        if stop_button_pressed:
            st.write("Processing stopped by user.")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
