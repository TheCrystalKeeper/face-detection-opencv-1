import cv2
import os
import imghdr

users_input = ''
possible_answers = ['1', '2']
while users_input not in possible_answers:
    users_input = input("Please enter the task number. \n 1 - Single Image Face Detection \n 2 - Webcam Face Detection\n")
    if users_input not in possible_answers:
        print("Bad input, please try again:")
if users_input == '1':
    # get filenames from user (also lists files in cwd)
    current_dir = os.getcwd()
    files_and_dirs = os.listdir(current_dir)

    # filter out directories, keeping only files
    files = [f for f in files_and_dirs if  (not os.path.isfile(os.path.join(current_dir, f))) or imghdr.what(f) in ['png', 'jpg', 'jpeg']]

    print("Choose a file. Pick one in the current working directory, or specify an absolute path1:")
    for file in files:
        if os.path.isfile(os.path.join(current_dir, file)):
            print(" - " +  file)
        else:
            print(" * " + file)
    image_filename = input("Enter filename here: ")

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
    print("unfinished")
