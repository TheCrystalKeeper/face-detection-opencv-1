import cv2
import os
import imghdr
import numpy as np

users_input = ''
possible_answers = ['1', '2', '3']
while users_input not in possible_answers:
    users_input = input("Please enter the task number. \n 1 - Single Image Face Detection \n 2 - Webcam Face Detection\n 3 - New Webcam Face Detection\n")
    if users_input not in possible_answers:
        print("Bad input, please try again:")
if users_input == '1':
    # get filenames from user (also lists files in cwd)
    current_dir = os.getcwd()
    files_and_dirs = os.listdir(current_dir)

    # filter out directories, keeping only files
    files = [f for f in files_and_dirs if  (not os.path.isfile(os.path.join(current_dir, f))) or imghdr.what(f) in ['png', 'jpg', 'jpeg']]

    print("Choose a file. Pick one in the current working directory, or specify an absolute path1:\n===============================================")
    for file in files:
        if os.path.isfile(os.path.join(current_dir, file)):
            print(" - " +  file)
        else:
            print(" * " + file)
    image_filename = input("=============================================== \nEnter filename here: ")

    # go to default image
    if image_filename == '':
        image_filename = 'Defaults/the-office.jpeg'

    image = cv2.imread(image_filename)
    if image is None:
        print("Error: Could not load image.")
    else:
        # convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # load the pre-trained Haar Cascade classifier for face detection
        haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # detect faces in the grayscale image
        faces_rects = haar_cascade_face.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        
        # print the number of faces found
        print(f"Number of faces detected: {len(faces_rects)} \n\n")

        # draw rectangles around the faces
        for (x, y, w, h) in faces_rects:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Detected Faces', image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
elif users_input == '2':
    print("hi my dear")
    cap = cv2.VideoCapture(1)
    haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    haar_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        #cv2.imshow("Frame", frame)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = cv2.flip(frame, 1)

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # load the pre-trained Haar Cascade classifier for face detection
        #haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # detect faces in the grayscale image
        #faces_rects = haar_cascade_face.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        faces_rects2 = haar_cascade_profile.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        faces_rects3 = haar_cascade_profile.detectMultiScale(cv2.flip(gray_image, 1), scaleFactor=1.1, minNeighbors=5)
        #faces_rects = faces_rects2 + faces_rects3
        
        # print the number of faces found
        #print(f"Number of faces detected: {len(faces_rects)} \n\n")

        # draw rectangles around the faces
        #for (x, y, w, h) in faces_rects:
         #   cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)
        for (x, y, w, h) in faces_rects2:
            cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)
        for (x, y, w, h) in faces_rects3:
            cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (255, 0, 0), 2)
        
        cv2.imshow('Detected Faces', frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

elif users_input == '3':
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        exit()

    model_file = 'res10_300x300_ssd_iter_140000.caffemodel'
    config_file = 'deploy_lowres.prototxt'
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        cv2.imshow('Detected Faces', frame)
        # check for esc
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
