import cv2
import face_recognition
import os

#define width anhd heigth of img after cropped
img_width_cropped=216
img_height_cropped=216

#getimg
imageraw_path="ImageRaw"
image_path="Images"

imgList=[]
for filename in os.listdir(imageraw_path):
    imgList.append(filename)

for filename in imgList:
    #get face location
    img_path=os.path.join(imageraw_path,filename)
    image=cv2.imread(img_path)
    image_height,image_width,_=image.shape
    
    #define face location
    faceCurFrame=face_recognition.face_locations(image)
    y1,x2,y2,x1=faceCurFrame[0]
    # print(x1,y1,x2,y2)
    bbox_width=x2-x1
    bbox_height=y2-y1
    bbox_max=max(bbox_width,bbox_height)
    
    #crop
    # crop_image = image[started_point_cropped_y:started_point_cropped_y+img_height_cropped,started_point_cropped_x:started_point_cropped_x+img_width_cropped]
    # cv2.imshow("Small Img", crop_image)
    # cv2.waitKey(0)129 751 0 622
    
    if x1<y1:
        inc_value=min(x1,image_height-(y1+bbox_max))
        crop_image = image[-inc_value+y1 : y1+bbox_max+inc_value, x1-inc_value:bbox_max+inc_value+x1]
        
    else:
        inc_value=min(y1,image_width-(x1+bbox_max)) 
        crop_image = image[y1-inc_value:bbox_max+inc_value+y1,x1-inc_value:x1+bbox_max+inc_value]
        
    current_width,_,_ = crop_image.shape
    

    # #resize img
    imgS = cv2.resize(crop_image, (0, 0), fx =img_width_cropped/current_width, fy = img_height_cropped/current_width)
    cv2.imshow("Small Img", imgS)
    cv2.waitKey(0)
    
    #save images
    cv2.imwrite(os.path.join(image_path,filename), imgS) 