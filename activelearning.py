import cv2
import face_recognition as fp
import numpy as np
from datetime import datetime
encodeList = []
cname = 1
nameslist = ['piyush','athaarva','a','b']

img = cv2.imread('images/')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
encode = fp.face_encodings(img)[0]
encodeList.append(encode)
#namelist.append()

AttendanceList = {}
def markAttendance(name,room):
    with open('Attendance.csv','a') as f:
        if AttendanceList.get(name,-1)!=room:
            AttendanceList[name]=room
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtString},{room}')

def findEncoding(img):
    global cname
    if len(fp.face_encodings(img))==0:
        return
    encode = fp.face_encodings(img)[0]
    encodeList.append(encode)
    name = 'P ' + str(cname)
    cname+=1
    nameslist.append(name)
    
# sources = ['rtsp://admin:InThink@2023@169.254.47.130:554/Streaming/Channels/101/','rtsp://admin:InThink@2023@169.254.47.130:554/Streaming/Channels/101/']
sources = [1,0]
caps = []
for i in sources:
    caps.append(cv2.VideoCapture(i))
 
while True:
    
    for room,cap in enumerate(caps):        
        success, img = cap.read()

        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        facesCurFrame = fp.face_locations(imgS)
        encodesCurFrame = fp.face_encodings(imgS,facesCurFrame)
    
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            # matches = fp.compare_faces(encodeList,encodeFace)
            faceDis = fp.face_distance(encodeList,encodeFace)

            y1,x2,y2,x1 = faceLoc
            faceCrop = imgS[y1:y2,x1:x2]
            
            matchIndex = np.argmin(faceDis)

            if faceDis[matchIndex]< 0.60:
                name = nameslist[matchIndex]
                markAttendance(name,room)
            else:
                findEncoding(faceCrop)
                print(encodeList)
                matchIndex = np.argmin(faceDis)
                name = nameslist[matchIndex]           
                print(name)
            
            # y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        
        cv2.imshow(str(room),img)
        cv2.waitKey(1)
    