import cv2 as cv
import os
import shutil


def main(input_file: str, directory: str):
    cap = cv.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    i = 0
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv.imwrite(os.path.join(directory, f'frame_{i}.jpg'), frame)
            i = i+1
        else:
            break
    cap.release()


if __name__ == '__main__':
    file = input('Enter input file:')
    directory = input('Enter directory:')
    main(file, directory)

