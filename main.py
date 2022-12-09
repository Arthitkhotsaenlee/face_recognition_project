from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import face_recognition

# Path DIR
images_DIR = "/Users/arthitkhotsaenlee/pythonProject/face_recognition_pea_project/images_db"
info_DIR = "/Users/arthitkhotsaenlee/pythonProject/face_recognition_pea_project/info_db"

global capture, rec_frame, grey, switch, neg, face, out
capture = 0
grey = 0
neg = 0
face = 0
switch = 1

# make shots directory to save pics
try:
    os.mkdir('./images_db')
except OSError as error:
    pass

# instatiate flask app
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)


# import know peoples
know_list = [i for i in os.listdir(images_DIR)]
known_face_encodings = []
known_face_names = []
for i in know_list:
    img = face_recognition.load_image_file(os.path.join(images_DIR,i))
    img_encode = face_recognition.face_encodings(img)[0]
    name = i.split(".")[0]
    known_face_encodings.append(img_encode)
    known_face_names.append(name)


def detect_face(frame):

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        # get the recognised name
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(str(name))

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            # put name to frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name[:-1], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if (face):
                frame = detect_face(frame)
            if (grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (neg):
                frame = cv2.bitwise_not(frame)
            if (capture):
                capture = 0
                now = "arthit"
                p = os.path.sep.join(['images_db', "{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face':
            global face
            face = not face
        elif request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1


    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()