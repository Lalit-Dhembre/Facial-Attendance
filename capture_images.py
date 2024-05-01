from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, Button, PhotoImage, messagebox
import cv2
import os
import albumentations as A
import numpy as np
import pandas as pd

class ImageCapture:
    def __init__(self, output_dir, name, roll_no):
        self.output_dir = output_dir
        self.name = name
        self.roll_no = roll_no

    def augment_image(self, image):
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
        ])
        
        images = [transform(image=image)['image'] for _ in range(3)]
        return images

    def capture_images(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, self.output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Unable to open the camera.")
            return

        num_images = 20
        student_info = {'Name': [], 'Roll No': []}
        images_folder = os.path.join(output_dir, self.name)
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        images_taken = 0
        while images_taken < num_images:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image.")
                break

            cv2.imshow('Capture Image', frame)
            image_path = os.path.join(images_folder, f'{self.name}_{images_taken + 1}.jpg')
            cv2.imwrite(image_path, frame)
            print(f"Original Image {images_taken + 1}/{num_images} captured for {self.name}")
            augmented_images = self.augment_image(frame)
            for i, aug_image in enumerate(augmented_images, start=1):
                aug_image_path = os.path.join(images_folder, f'{self.name}_{images_taken + 1}_aug_{i}.jpg')
                cv2.imwrite(aug_image_path, aug_image)
                print(f"Augmented Image {i} for {self.name}_{images_taken + 1} saved.")

            images_taken += 1

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        student_info['Name'].append(self.name)
        student_info['Roll No'].append(self.roll_no)
        df = pd.DataFrame(student_info)
        df['Roll No'] = pd.to_numeric(df['Roll No'])
        df.sort_values(by='Roll No', inplace=True)
        excel_path = 'student_info.xlsx'
        if os.path.exists(excel_path):
            existing_df = pd.read_excel(excel_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            df.drop_duplicates(subset=['Roll No'], keep='first', inplace=True)
        else:
            pass

        df.sort_values(by='Roll No', inplace=True)
        df.to_excel(excel_path, index=False)
        subjects = ['CN', 'COA', 'AT', 'MATHS', 'OS']
        for subject in subjects:
            subject_excel_path = os.path.join('subject_attendance', f'{subject}_attendance.xlsx')
            if os.path.exists(subject_excel_path):
                subject_df = pd.read_excel(subject_excel_path)
            else:
                subject_df = pd.DataFrame(columns=['Name', 'Roll No'])
            subject_df = pd.concat([subject_df, df], ignore_index=True)
            subject_df.drop_duplicates(subset=['Roll No'], keep='first', inplace=True)
            subject_df.sort_values(by='Roll No', inplace=True)
            subject_df.to_excel(subject_excel_path, index=False)

if __name__ == "__main__":
    image_capture = ImageCapture()
    image_capture.capture_images()