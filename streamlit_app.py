import streamlit as st
import cv2
from ultralytics import YOLO
import os
import shutil

# Specify the path of the folder to be deleted
folder_path = 'runs'


def main():
    st.title("Image Upload and Processing")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

    if uploaded_file is not None:
        # Save the image to a folder
        image_path = os.path.join("static/uploads", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Perform YOLO object detection
        model = YOLO("best.pt")
        model.predict(image_path, save=True)

        # Load the processed image
        processed_image_path = os.path.join(
            "runs/detect/predict", uploaded_file.name)
        img = cv2.imread(processed_image_path)

        # Save the processed image
        output_image_path = os.path.join("static/results", uploaded_file.name)
        cv2.imwrite(output_image_path, img)

        # Remove the 'runs' folder and its contents
        shutil.rmtree(folder_path, ignore_errors=True)

        # Display the processed image
        st.image(output_image_path, caption="Processed Image")


if __name__ == "__main__":
    main()
