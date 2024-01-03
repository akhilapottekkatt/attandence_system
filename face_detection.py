import cv2
import numpy as np
import face_recognition as face_reg

# Function
def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

# Image declaration
akhila = face_reg.load_image_file('sample/akhila.jpg')
akhila = cv2.cvtColor(akhila, cv2.COLOR_BGR2RGB)
akhila = resize(akhila, 0.50)

akhila_test = face_reg.load_image_file('sample/lal.jpg')
akhila_test = resize(akhila_test, 0.50)
akhila_test = cv2.cvtColor(akhila_test, cv2.COLOR_BGR2RGB)

# Finding face locations
facelocation_akhila = face_reg.face_locations(akhila)
if not facelocation_akhila:
    print("No face found in the training image.")
else:
    facelocation_akhila = facelocation_akhila[0]
    encode_akhila = face_reg.face_encodings(akhila, [facelocation_akhila])[0]
    cv2.rectangle(akhila, (facelocation_akhila[3], facelocation_akhila[0]),
                  (facelocation_akhila[1], facelocation_akhila[2]), (255, 0, 255), 3)

facelocation_akhila_test = face_reg.face_locations(akhila_test)
if not facelocation_akhila_test:
    print("No face found in the test image.")
else:
    facelocation_akhila_test = facelocation_akhila_test[0]
    encode_akhila_test = face_reg.face_encodings(akhila_test, [facelocation_akhila_test])[0]
    cv2.rectangle(akhila_test, (facelocation_akhila_test[3], facelocation_akhila_test[0]),
                  (facelocation_akhila_test[1], facelocation_akhila_test[2]), (255, 0, 255), 3)

    # Compare faces
    results = face_reg.compare_faces([encode_akhila], encode_akhila_test)
    print("Face match result:", results)

cv2.imshow('train_img', akhila)
cv2.imshow('test_img', akhila_test)
cv2.waitKey(0)
cv2.destroyAllWindows()


# def resize(img, size):
#     width = int(img.shape[1]*size)
#     height = int(img.shape[0]*size)
#     dimension = (width, height)
#     return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
#
# # img declaration
# akhila = face_reg.load_image_file('sample/akhila.jpg')
# akhila = cv2.cvtColor(akhila, cv2.COLOR_BGR2RGB)
# akhila = resize(akhila, 0.50)
# akhila_test = face_reg.load_image_file('sample/akhila_text.jpg')
# akhila_test = resize(akhila_test, 0.50)
# akhila_test = cv2.cvtColor(akhila_test, cv2.COLOR_BGR2RGB)
#
# #finding face locations
#
# facelocation_akhila = face_reg.face_locations(akhila)[0]
# encode_akhila = face_reg.face_encodings(akhila)[0]
# cv2.rectangle(akhila, (facelocation_akhila[3], facelocation_akhila[0]), (facelocation_akhila[1], facelocation_akhila[2]), (255,0,255),3)
#
# facelocation_akhila_test = face_reg.face_locations(akhila_test)[0]
# encode_akhila_test = face_reg.face_encodings(akhila_test)[0]
# cv2.rectangle(akhila_test, (facelocation_akhila[3], facelocation_akhila[0]), (facelocation_akhila[1], facelocation_akhila[2]), (255,0,255),3)
#
# results = face_reg.compare_faces([encode_akhila], encode_akhila_test)
#
#
# cv2.imshow('train_img', akhila)
# cv2.imshow('test_img', akhila_test)
# cv2.waitKey(0)
# cv2.destroyWindow()
