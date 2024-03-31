import cv2
import os
import albumentations as A
import numpy as np
import pandas as pd

def augment_image(image):
    # Define an augmentation pipeline
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),  # Randomly adjust brightness and contrast
        A.RandomGamma(p=0.5),  # Randomly adjust gamma
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),  # Simulate zoom in/out
    ])
    
    # Apply the transformations 3 times to create 3 variations
    images = [transform(image=image)['image'] for _ in range(3)]
    return images

def capture_images(output_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return

    num_images = int(input("Enter the number of images to capture for each person: "))

    # Creating DataFrame to store student information
    student_info = {'Name': [], 'Roll No': []}

    while True:
        name = input("Enter the name for the person (or 'q' to quit): ")
        if name.lower() == 'q':
            break

        if name.strip() == '':
            print('Name cannot be empty. Please try again.')
            continue

        roll_no = input("Enter the roll number for the person: ")
        if roll_no.strip() == '':
            print('Roll number cannot be empty. Please try again.')
            continue

        images_folder = os.path.join(output_dir, name)
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        images_taken = 0
        while images_taken < num_images:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image.")
                break

            cv2.imshow('Capture Image', frame)

            # Save the original image
            image_path = os.path.join(images_folder, f'{name}_{images_taken + 1}.jpg')
            cv2.imwrite(image_path, frame)
            print(f"Original Image {images_taken + 1}/{num_images} captured for {name}")

            # Generate and save augmented images
            augmented_images = augment_image(frame)
            for i, aug_image in enumerate(augmented_images, start=1):
                aug_image_path = os.path.join(images_folder, f'{name}_{images_taken + 1}_aug_{i}.jpg')
                cv2.imwrite(aug_image_path, aug_image)
                print(f"Augmented Image {i} for {name}_{images_taken + 1} saved.")

            images_taken += 1

            if cv2.waitKey(1) == ord('q'):
                break

        # Append student info to DataFrame
        student_info['Name'].append(name)
        student_info['Roll No'].append(roll_no)

    cap.release()
    cv2.destroyAllWindows()

    # Creating DataFrame from student_info dictionary
    df = pd.DataFrame(student_info)
    # Sorting DataFrame by Roll No
    df.sort_values(by='Roll No', inplace=True)
    # Writing DataFrame to a single Excel file
    df.to_excel('student_info.xlsx', index=False)

    # Creating Excel files for each subject
    subjects = ['CN','COA','AT','MATHS','OS']  # Example list of subjects
    for subject in subjects:
        subject_df = df.copy()  # Create a copy of the DataFrame
        subject_df.to_excel(os.path.join('subject_attendance', f'{subject}_attendance.xlsx'),index=False)


def main():
    output_dir = 'Faces'  # Specific directory to store images
    capture_images(output_dir)

if __name__ == "__main__":
    main()