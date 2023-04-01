import cv2
import numpy as np
from PIL import Image 
import pickle
import os
from datetime import datetime

'''
names = []
filename = "names.pkl"
f = open(filename, 'wb')
pickle.dump(names,f)
f.close()
'''

nameList = {}
def markAttendance(name):
    with open('Attendance.csv','a') as f:
        if nameList.get(name,0)==0:
            nameList[name]=1
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtString}')

def faceSampling():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) 
    cam.set(4, 480) 

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)

    name = input('Enter name for the Face: ')
    names.append(name)
    id = names.index(name)

    print('Reading Face......')
    
    count = 0

    while(True):

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            
            cv2.imwrite("dataset/"+name+"." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff 
        if k == 27:
            break
        elif count >= 200: 
             break

    with open('names.pkl', 'wb') as f:
        pickle.dump(names, f)
    
    print("Your Face has been registered as {}\n\nExiting Sampling Program".format(name.upper()))
    cam.release()
    cv2.destroyAllWindows()

def faceLearning():
    
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create() 
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    print ("\nTraining for the faces has been started. It might take a while.\n")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    
    recognizer.write('trainer/trainer.yml') 

    print("{0} faces trained. Exiting Training Program".format(len(np.unique(ids))))

def faceRecognition():
    print('\nStarting Recognizer....')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

    
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) 
    cam.set(4, 480) 

    
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)

    while True:

        ret, img =cam.read()

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                markAttendance(id)
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  


        cv2.imshow('camera',img) 

        k = cv2.waitKey(10) & 0xff 
        if k == 27:
            break

    print("\nExiting Recognizer.")
    cam.release()
    cv2.destroyAllWindows()

def main():
    while True:
        choice = int(input('Enter \n1. New Face \n2. Face Recognition \n3. Exit\n'))
        
        if choice == 1:
            faceSampling()
            faceLearning()
        elif choice == 2:
            faceRecognition()
        elif choice == 3:
            print('\nExiting Program.')
            break

main()