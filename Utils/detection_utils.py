from PIL import Image
from faced import FaceDetector
from faced.utils import annotate_image
import cv2
from resizeimage import resizeimage
from Utils.fr_utils import *
from Utils.FR_UtilsV2 import *
from Utils.database import Add_to_Database,Remove_from_Database

def Detection_Faces(img_path):
    face_detector = FaceDetector()
    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    thresh=0.85
    bboxes = face_detector.predict(rgb_img, thresh)
    return bboxes

def FaceDetection_to_Cv2Rectangles(bboxes):

    if not bboxes:
        return "Nothing"
    i=0
    x=bboxes[0][0]
    y=bboxes[0][1]
    w=bboxes[0][2]
    h=bboxes[0][3]
    StartingPoint_x=int(x - w/2)
    StartingPoint_y= int(y - h/2)
    FinalPoint_x=int(x + w/2)
    FinalPoint_y=int(y + h/2)
    Color=(0, 255, 0)
    thickness=3


    return StartingPoint_x,StartingPoint_y,FinalPoint_x,FinalPoint_y,Color,thickness


def Crop_Image_to_Face(img_path,i):
    img = Image.open(img_path)
    bboxes=Detection_Faces(img_path)
    StartingPoint_x,StartingPoint_y,FinalPoint_x,FinalPoint_y,Color,thickness=FaceDetection_to_Cv2Rectangles(bboxes)
    img2 = img.crop((StartingPoint_x, StartingPoint_y, FinalPoint_x, FinalPoint_y))
    im = img2.resize((96,96),Image.ANTIALIAS)
    im.save("Images/WebCamResize/Face_Resize_"+str(i)+".jpg")
    return "Images/WebCamResize/Face_Resize_"+str(i)+".jpg"

def Video_Detection_Recognition(database,FRmodel):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    Photo_Dict={}
    Identity_Array=[]
    img_counter = 0

    while True:
        camaraOn=True
        i=0
        ret, frame = cam.read()
        Name_Placer = "Wait..."
        font=cv2.FONT_HERSHEY_SIMPLEX
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'',(470,70),font,1,(0,0,0),2)

        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k%256 == 32:
            while camaraOn==True:
                i=i+1

                if i >1:
                  cv2.rectangle(frame, (StartingPoint_x, StartingPoint_y), (FinalPoint_x, FinalPoint_y),
                          Color, thickness)
                  cv2.putText(frame,Name_Placer,(470,70),font,1,(0,0,0),2)
                  cv2.imshow("test", frame)

                a = cv2.waitKey(1)
                cv2.waitKey(10)
                ret, frame = cam.read()
                img_name = "Images/WebCamDetection/opencv_frame_{}.jpg".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1

                with open('Images/WebCamDetection/opencv_frame_'+str(i)+'.jpg', 'r+b') as f:
                    with Image.open(f) as image:
                        image_path = 'Images/WebCamDetection/opencv_frame_'+str(i)+'.jpg'
                        image.save(image_path, image.format)
                        bboxes = Detection_Faces(image_path)
                        if not bboxes:
                            StartingPoint_x=0
                            StartingPoint_y=0
                            FinalPoint_x=0
                            FinalPoint_y=0
                            Color=0
                            thickness=0
                            Name_Placer="No one..."
                            cv2.putText(frame,Name_Placer,(470,70),font,1,(0,0,0),2)
                            cv2.imshow("test", frame)
                        else:
                            StartingPoint_x,StartingPoint_y,FinalPoint_x,FinalPoint_y,Color,thickness=FaceDetection_to_Cv2Rectangles(bboxes)
                            Crop_Img_Location=Crop_Image_to_Face(image_path,i)
                            min_dist , identity =who_is_it(Crop_Img_Location, database, FRmodel)
                            if min_dist > 0.7:
                                Name_Placer="Not Known"
                                cv2.putText(frame,Name_Placer,(470,70),font,1,(0,0,0),2)
                                cv2.imshow("test", frame)
                                print("Not in the database.")
                                print(min_dist)
                            else:
                                print ("it's " + str(identity) + ", the distance is " + str(min_dist))
                                Name_Placer=identity
                if a%256 == 27:
                    camaraOn=False
                    break
                ret, frame = cam.read()
                cv2.imshow("test", frame)
    cam.release()

    cv2.destroyAllWindows()


def Image_Detection_Recognition(image_path,database,FRmodel):
    img = cv2.imread(image_path)
    image = Image.open(image_path)
    image.save(image_path)
    bboxes = Detection_Faces(image_path)
    font=cv2.FONT_HERSHEY_SIMPLEX

    if not bboxes:
            StartingPoint_x=0
            StartingPoint_y=0
            FinalPoint_x=0
            FinalPoint_y=0
            Color=0
            thickness=0
            Name_Placer="No one..."
            ann_img = annotate_image(img, bboxes)
            cv2.putText(ann_img,Name_Placer,(470,70),font,1,(0,0,0),2)
            cv2.imshow("test", ann_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:

        ann_img = annotate_image(img, bboxes)
        StartingPoint_x,StartingPoint_y,FinalPoint_x,FinalPoint_y,Color,thickness=FaceDetection_to_Cv2Rectangles(bboxes)
        Crop_Image_to_Face(image_path,i)
        min_dist , identity =who_is_it(image_path, database, FRmodel)
        if min_dist > 0.96:
            Name_Placer="Not Known"
            cv2.putText(ann_img,Name_Placer,(470,70),font,1,(0,0,0),2)
            cv2.imshow("test", ann_img)
            print("Not in the database.")
            print(min_dist)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print ("it's " + str(identity) + ", the distance is " + str(min_dist))
            Name_Placer=identity
            cv2.putText(ann_img,Name_Placer,(470,70),font,1,(0,0,0),2)
            cv2.imshow("test", ann_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
