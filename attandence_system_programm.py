import os
import cv2
import numpy as np
import face_recognition as face_reg
from datetime import datetime
import pyttsx3 as textspeech

engine = textspeech.init()
engine.setProperty('rate', 150)  # Adjust the rate as needed

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

path = "students_imgs"

# Check if the directory exists
if not os.path.exists(path):
    print(f"Error: The directory '{path}' does not exist.")
    exit()

studentimg = []
studentName = []
mylist = os.listdir(path)
print(mylist)

if not mylist:
    print(f"Error: No files found in the directory '{path}'.")
    exit()

for cl in mylist:
    curImg = cv2.imread(os.path.join(path, cl))

    # Check if the image is successfully loaded
    if curImg is None:
        print(f"Error: Failed to load image '{cl}' from the directory '{path}'.")
        continue

    studentimg.append(curImg)
    studentName.append(os.path.splitext(cl)[0])

print(studentName)


def finEncoding(images):
    imgencodings = []
    for img in images:
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_reg.face_encodings(img)

        # Check if a face is found in the image
        if not encodeimg:
            print("Error: No face found in the image.")
            continue

        imgencodings.append(encodeimg[0])
    return imgencodings


def MarkAttendance(name):
    file_path = 'attendance.csv'

    # Check if the file exists, create it if it doesn't
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('Name, Time\n')

    # Open the file in read and write mode
    with open(file_path, 'r+') as f:
        myDataList = f.read()
        nameList = [entry.split(',')[0] for entry in myDataList.split('\n') if entry]

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.write(f'\n{name}, {timestr}')
            engine.say("Welcome to class " + name)

Encodelist = finEncoding(studentimg)

vid = cv2.VideoCapture(0)

if not vid.isOpened():
    print("Error: Failed to open the camera.")
    exit()

while True:
    success, frame = vid.read()
    if not success:
        print("Error: Failed to read frame from the camera.")
        break

    # Resize the frame before face detection
    frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    # Use a lower value for number_of_times_to_upsample
    facesinframe = face_reg.face_locations(frames, number_of_times_to_upsample=1)
    encodeinframe = face_reg.face_encodings(frames, facesinframe)

    for encodeface, faceloc in zip(encodeinframe, facesinframe):
        matches = face_reg.compare_faces(Encodelist, encodeface)
        facedis = face_reg.face_distance(Encodelist, encodeface)

        # Keep the threshold at 0.3
        if matches and min(facedis) < 0.6:
            matchIndex = np.argmin(facedis)
            name = studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendance(name)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()





# import os
# import cv2
# import numpy as np
# import face_recognition as face_reg
# from datetime import datetime
# import pyttsx3 as textspeech
#
#
# engine = textspeech.init()
# engine.setProperty('rate', 150)  # Adjust the rate as needed
#
# def resize(img, size):
#     width = int(img.shape[1] * size)
#     height = int(img.shape[0] * size)
#     dimension = (width, height)
#     return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
#
#
# path = "students_imgs"
#
# # Check if the directory exists
# if not os.path.exists(path):
#     print(f"Error: The directory '{path}' does not exist.")
#     exit()
#
# studentimg = []
# studentName = []
# mylist = os.listdir(path)
# print(mylist)
#
# if not mylist:
#     print(f"Error: No files found in the directory '{path}'.")
#     exit()
#
# for cl in mylist:
#     curImg = cv2.imread(os.path.join(path, cl))
#
#     # Check if the image is successfully loaded
#     if curImg is None:
#         print(f"Error: Failed to load image '{cl}' from the directory '{path}'.")
#         continue
#
#     studentimg.append(curImg)
#     studentName.append(os.path.splitext(cl)[0])
#
# print(studentName)
#
#
# def finEncoding(images):
#     imgencodings = []
#     for img in images:
#         img = resize(img, 0.50)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encodeimg = face_reg.face_encodings(img)
#
#         # Check if a face is found in the image
#         if not encodeimg:
#             print("Error: No face found in the image.")
#             continue
#
#         imgencodings.append(encodeimg[0])
#     return imgencodings
#
#
# def MarkAttendance(name):
#     file_path = 'attendance.csv'
#
#     # Check if the file exists, create it if it doesn't
#     if not os.path.exists(file_path):
#         with open(file_path, 'w') as f:
#             f.write('Name, Time\n')
#
#     # Open the file in read and write mode
#     with open(file_path, 'r+') as f:
#         myDataList = f.read()
#         nameList = [entry.split(',')[0] for entry in myDataList.split('\n') if entry]
#
#         if name not in nameList:
#             now = datetime.now()
#             timestr = now.strftime('%H:%M')
#             f.write(f'\n{name}, {timestr}')
#             engine.say("Welcome to class " + name)
#
# Encodelist = finEncoding(studentimg)
#
# vid = cv2.VideoCapture(0)
#
# if not vid.isOpened():
#     print("Error: Failed to open the camera.")
#     exit()
#
# while True:
#     success, frame = vid.read()
#     if not success:
#         print("Error: Failed to read frame from the camera.")
#         break
#
#     frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
#
#     facesinframe = face_reg.face_locations(frames)
#     encodeinframe = face_reg.face_encodings(frames, facesinframe)
#
#     for encodeface, faceloc in zip(encodeinframe, facesinframe):
#         matches = face_reg.compare_faces(Encodelist, encodeface)
#         facedis = face_reg.face_distance(Encodelist, encodeface)
#
#         # Adjust the threshold as needed (e.g., 0.5, 0.7, etc.)
#         if matches and min(facedis) < 0.3:
#             matchIndex = np.argmin(facedis)
#             name = studentName[matchIndex].upper()
#             y1, x2, y2, x1 = faceloc
#             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#             MarkAttendance(name)
#
#     cv2.imshow('video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# vid.release()
# cv2.destroyAllWindows()
#




# import os
# import cv2
# import numpy as np
# import face_recognition as face_reg
# from datetime import datetime
# def resize(img, size):
#     width = int(img.shape[1] * size)
#     height = int(img.shape[0] * size)
#     dimension = (width, height)
#     return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
#
# path = "students_imgs"
# studentimg = []
# studentName = []
# mylist = os.listdir(path)
# print(mylist)
#
# for cl in mylist:
#     curImg = cv2.imread(os.path.join(path, cl))  # Use os.path.join for correct file path
#     studentimg.append(curImg)
#     studentName.append(os.path.splitext(cl)[0])
#
# print(studentName)
#
# def finEncoding(images):
#     imgencodings = []
#     for img in images:
#         img = resize(img, 0.50)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encodeimg = face_reg.face_encodings(img)[0]
#         imgencodings.append(encodeimg)
#     return imgencodings
#
# def MarkAttendence(name):
#     with open('attandance.csv', 'r+',) as f:
#         myDataList = f.readlines()
#         nameList =[]
#         for line in myDataList :
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList :
#             now = datetime.now()
#             timestr =now.strftime('%H: %M')
#             f.writelines(f'\n{name}, {timestr}')
#
#
#
# Encodelist = finEncoding(studentimg)
#
# vid = cv2.VideoCapture(0)
#
# while True:
#     success, frame = vid.read()
#     if not success:
#         print("Error: Failed to read frame from the camera.")
#         break  # Exit the loop or handle the error appropriately
#     frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
#
#     facesinframe = face_reg.face_locations(frames)
#     encodeinframe = face_reg.face_encodings(frames, facesinframe)
#
#     for encodeface, faceloc in zip(encodeinframe, facesinframe):
#         matches = face_reg.compare_faces(Encodelist, encodeface)
#         facedis = face_reg.face_distance(Encodelist, encodeface)
#         print(facedis)
#         matchIndex = np.argmin(facedis)
#
#         if matches[matchIndex]:
#             name = studentName[matchIndex].upper()
#             y1, x2, y2, x1 = faceloc
#             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#             MarkAttendence(name)
#
#     cv2.imshow('video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# vid.release()
# cv2.destroyAllWindows()
#
#
#
#
