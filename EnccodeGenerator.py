import cv2
import face_recognition
import os
import pickle

# Import Images
path_img='ImageRaw'
imgList=[]
imgID=[]
for path in os.listdir(path_img):
    imgList.append(cv2.imread(os.path.join(path_img,path)))
    imgID.append(os.path.splitext(path)[0])
# print(imgID)

#opencv use BGR, facerec use RGB
def findEncodings(imgList):
    encodeList = []
    for img in imgList:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList

print("Encoding started....")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithID=[encodeListKnown,imgID]
print("Encoding finished")

file=open("EncodeFile.p","wb")
pickle.dump(encodeListKnownWithID,file)
file.close()
print("File saved")
