import cv2
import numpy as np
import face_recognition

imgap = face_recognition.load_image_file('mainImage.jpg')
imgap = cv2.cvtColor(imgap,cv2.COLOR_BGR2RGB)

imgap_test = face_recognition.load_image_file('testImage.jpg')
imgap_test = cv2.cvtColor(imgap_test,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgap)[0]
encodeap = face_recognition.face_encodings(imgap)[0]
cv2.rectangle(imgap,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgap_test)[0]
encodeapTest = face_recognition.face_encodings(imgap_test)[0]
cv2.rectangle(imgap_test,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodeap],encodeapTest)
faceDis = face_recognition.face_distance([encodeap],encodeapTest)
print(result," ",faceDis)
cv2.putText(imgap_test,f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Main Image',imgap)
cv2.imshow('Test Image',imgap_test)
cv2.waitKey(0)