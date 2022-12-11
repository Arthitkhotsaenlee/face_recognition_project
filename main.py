from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import numpy as np
import json
import face_recognition
import uuid
import pandas as pd

# Path DIR
"""
 Here is the path directory session.
 User may chane it in to sql IP address.
"""
images_DIR = "images_db"
info_DIR = "info_db"

capture = False
face = False

# make shots directory to save pics

os.makedirs(images_DIR,exist_ok=True)
os.makedirs(info_DIR,exist_ok=True)


# instatiate flask app
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)


# import know peoples
know_list = [i for i in os.listdir(images_DIR)]
known_face_encodings = []
known_face_names = []
for i in know_list:
    try:
        img = []
        img = face_recognition.load_image_file(os.path.join(images_DIR, i))
        img_encode = face_recognition.face_encodings(img)[0]
        name = i.split(".")[0]
        known_face_encodings.append(img_encode)
        known_face_names.append(name)
    except Exception:
        pass

# import know information from database
know_information = []
try:
    with open("info_db/information_db.json", "r") as openfile:
        # Reading from json file
        json_info = json.load(openfile)
    know_information = pd.DataFrame.from_dict(json_info)
except Exception:
    pass


def write_information(data_df):
    try:
        global know_information
        # Opening JSON file
        with open("info_db/information_db.json", "r") as openfile:
            # Reading from json file
            json_object = json.load(openfile)
        information_df = pd.DataFrame.from_dict(json_object)
        information_df = pd.concat([information_df,data_df], ignore_index=True)
        information_df.to_json(path_or_buf=os.path.join(info_DIR, "information_db.json"))
        know_information = information_df
    except Exception:
        know_information = data_df
        data_df.to_json(path_or_buf=os.path.join(info_DIR, "information_db.json"))


def detect_face(frame,known_face_encodings,known_face_names):
    resize_param = 6
    ratio = 1/resize_param
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=ratio, fy=ratio)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        try:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(str(name))
        except Exception:
            face_names.append("Unknown")
        # get the recognised name

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= resize_param
        right *= resize_param
        bottom *= resize_param
        left *= resize_param
        # Draw a box around the face
        frame = cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        frame = cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        # put name to frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        try:
            mat_name = know_information[know_information["id"] == name]["fname"].values[0]
        except Exception:
            mat_name = "Unknow"
        frame = cv2.putText(frame, mat_name, (left +12, bottom - 6), font, 1.0, (255, 255, 255), 3)

    return frame


def check_info(frame):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        try:
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
        except Exception:
            face_names.append(["Unknow"])
        # Display the results
    info_df = pd.DataFrame()
    for kname in face_names:
        if kname != "Unknown":
            info_df = pd.concat([info_df,know_information[know_information["id"] == kname]])
    return info_df


def gen_frames():  # generate frame by frame from camera
    global capture, known_face_encodings, known_face_names
    while True:
        success, frame = camera.read()
        if success:
            if (face):
                frame = detect_face(frame,known_face_encodings,known_face_names)
            elif (capture):
                capture = 0
                file_name = primary_key
                image_name = "{}.png".format(str(file_name))
                p = os.path.sep.join(['images_db', image_name])
                cv2.imwrite(p, frame)
                pic1 = frame
                pic_encode1 = face_recognition.face_encodings(pic1)[0]
                name1 = file_name
                known_face_encodings.append(pic_encode1)
                known_face_names.append(name1)

            try:
                # show streaming video
                ret, buffer = cv2.imencode('.jpg',frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception:
                pass

@app.route('/')
def index():
    return render_template('face_recognition.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST', 'GET'])
def register():
    global camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture, primary_key
            primary_key = str(uuid.uuid4())
            reg_dict = {
                "id": [primary_key],
                "fname":[request.form.get("fname")],
                "lname": [request.form.get("lname")],
                "bday": [request.form.get("bday")],
                "email": [request.form.get("email")],
                "phone": [request.form.get("phone")],
            }
            if len(reg_dict["fname"]) > 0:
                reg_df = pd.DataFrame.from_dict(reg_dict)
                write_information(reg_df)
                capture = 1
        elif request.form.get('back') == 'Back':
            return redirect("/")
    elif request.method == 'GET':
        return render_template('register.html')
    return render_template('register.html')


@app.route("/checkInfomation", methods=['POST', 'GET'])
def checkInfomation():
    map_columna_name = {
        "id": "UserID",
        "fname": "First Name",
        "lname": "Last Name",
        "email": "Email",
        "phone": "Phone number"
    }
    if request.method == 'POST':
        if request.form.get('back') == 'Back':
            return redirect("/")
    elif request.method == 'GET':
        if not result_check.empty:
            show_info_dict = result_check.rename(map_columna_name,axis="columns")
        else:
            show_info_dict = pd.DataFrame.from_dict({"Unknown": ["Please add data to the database or capture a clearer photo"]})
        return render_template("chekinfo.html", tables=[show_info_dict.to_html()], titles=[''])
    if not result_check.empty:
        show_info_dict = result_check.rename(map_columna_name,axis="columns")
    else:
        show_info_dict = pd.DataFrame.from_dict({"Unknown": ["Please add data to the database or capture a clearer photo"]})
    return render_template("chekinfo.html", tables=[show_info_dict.to_html()], titles=[''])



@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global camera
    if request.method == 'POST':

        if request.form.get('face') == 'Face Recognition':
            global face
            face = not face

        elif request.form.get('regis') == 'Register':
            face = False
            return redirect(url_for("register"))

        elif request.form.get('info') == 'Get Information':
            face = False
            global result_check
            success, frame = camera.read()
            if success:
                result_check = check_info(frame)
            return redirect(url_for("checkInfomation"))

    elif request.method == 'GET':
        return render_template('face_recognition.html')
    return render_template('face_recognition.html')


if __name__ == '__main__':
    app.run()
camera.release()
cv2.destroyAllWindows()