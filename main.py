import cv2
import os
import face_recognition
import pickle
import numpy as np
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3,640)#thông số của x cam
cap.set(4,480)#thông số của y cam

#Background
imgBackGround = cv2.imread("Resources/background.jpg")
# imgBackGround=cv2.resize(imgBackGround,(1280,720))

#Import mode
path_mode='Resources/Modes'
imgModesList=[]
for path in os.listdir(path_mode):
    imgModesList.append(cv2.imread(os.path.join(path_mode,path)))

# Load Encoding Files
print("Loading Encoding Files...") 
file=open("EncodeFile.p","rb")
encodeListKnownWithID=pickle.load(file)
file.close()
encodeListKnown,imgID=encodeListKnownWithID
print("Encode Files Loaded") 

while True:
    success,img = cap.read()
    img = cv2.flip(img, 1)
    
    imgSmall =cv2.resize(img,(0,0),None,0.25,0.25)#rescale to reduce computation
    imgSmall=cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    
    faceCurFrame=face_recognition.face_locations(imgSmall)#Face location
    encodeCurFrame=face_recognition.face_encodings(imgSmall,faceCurFrame)#to compare data vs reality
    
    #Overlay webcam and background
    h,w,_=imgModesList[2].shape
    imgBackGround[140:140+480,55:55+640]=img #original point at top left rectangle
    imgBackGround[0:h,764:764+w]=imgModesList[2] 
    # imgBackGround[180:180+480,55:55+640]=img[1] 
    # imgBackGround[180:180+480,55:55+640]=img[2] 
    # imgBackGround[180:180+480,55:55+640]=img[3] 
    
    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):#Zip to combine 2 loops to 1 loops
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)#boolen if matches
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)#distance lower is better( it means may have more than 1 True img)
        print("Matches\n",matches,"Face distance",faceDis)
        
        matchIndex=np.argmin(faceDis)#index of true match
        
        if matches[matchIndex]:
            # print("Known Face Detected")
            # print(imgID[matchIndex])
            # cv2.rectangle(imgID[matchIndex],)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4 ,x2*4 ,y2*4 ,x1*4
            bbox=55+x1,140+y1, x2-x1, y2-y1
            imgBackGround=cvzone.cornerRect(imgBackGround,bbox,rt=0)#rt= rectangle thickness, rect detect face
    
    
    
    
    # cv2.imshow("Webcam",img) #Name of UI
    cv2.imshow("Face Recog",imgBackGround)
    # cv2.waitKey(1)#milisecond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
cv2.destroyAllWindows() 
