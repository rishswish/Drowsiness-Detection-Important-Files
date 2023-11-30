import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
import time


# counter=0
# def reset_counter():
#     global counter
#     counter=0

html_string = """
            <audio controls autoplay>
            <source src="https://www.orangefreesounds.com/wp-content/uploads/2022/04/Small-bell-ringing-short-sound-effect.mp3" type="audio/mp3">
            </audio>
            """

model = None
confidence = .25


def infer_image_eye(img, size=None):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_eye_everyone.pt', force_reload=True)
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image

def infer_image_yawn(img, size=None):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_laugh_yawn_dark.pt', force_reload=True)
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image

img_or_vid=st.sidebar.selectbox('Input as image or Input as Video?',('Image','Video'))

if img_or_vid=='Image':
    st.title('Take a Picture')
    runs = st.checkbox('Runs')
    while runs:
        # picture = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        picture = st.camera_input("Take a picture",key=1)
        if picture:
            pics=Image.open(picture)
            pics_array=np.array(pics)

            col1, col2 = st.columns(2)
            with col1:
                st.image(picture, caption="Selected Image")
            with col2:
                img_eye = infer_image_eye(pics_array)
                img_yawn=infer_image_yawn(pics_array)
            # img=cv2.addWeighted(src1=img_eye,alpha=0.5,src2=img_yawn,beta=0.5,gamma=0)
                st.image(img_yawn, caption="Model prediction")

yawny=0
closy=0
if img_or_vid=='Video':
    
    # video = st.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    model_eye = torch.hub.load('ultralytics/yolov5', 'custom', path='best_eye_everyone.pt', force_reload=True)
    model_yawn = torch.hub.load('ultralytics/yolov5', 'custom', path='best_laugh_yawn_dark.pt', force_reload=True)
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while run:  
        
        ret, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        roi=face.detectMultiScale(frame,1.3,5)
        blurred_frame=cv2.medianBlur(frame,45)

        for x,y,w,h in roi:
            detected_face=frame[int(y):int(y+h),int(x):int(x+w)]
            blurred_frame[y:y+h,x:x+w]=detected_face

        frame=blurred_frame
        results = model_eye(frame)
        results_y = model_yawn(np.squeeze(results.render()))
        FRAME_WINDOW.image(np.squeeze(results_y.render()))


        #Alarm for Yawn
        if len(results.xywh[0])>=1 :

                yawn_dconf=results_y.xywh[0][0][4]
                yawn_class=results_y.xywh[0][0][5]            

                l_eye_dconf=results.xywh[0][0][4]

                l_eye_class=results.xywh[0][0][5]

                if (yawn_dconf.item()>0.65 and yawn_class.item()==0.0) or ((l_eye_class.item()==1.0 or l_eye_class.item()!=0.0) and l_eye_dconf>0.5) :
                    yawny+=1
                    if yawny>=15:
                        sound = st.empty()
                        sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                        time.sleep(2)  # wait for 2 seconds to finish the playing of the audio
                        sound.empty()  # optionally delete the element afterwards
                    

                else:
                    yawny=0
        
        # if len(results.xywh[0])>=1 and len(results_y.xywh[0])==1:

        #         yawn_dconf=results_y.xywh[0][0][4]
        #         yawn_class=results_y.xywh[0][0][5]            

        #         l_eye_dconf=results.xywh[0][0][4]

        #         l_eye_class=results.xywh[0][0][5]

        #         if (yawn_dconf.item()>0.65 and yawn_class.item()==0.0) or ((l_eye_class.item()==1.0 or l_eye_class.item()!=0.0) and l_eye_dconf>0.55) :
        #             yawny+=1
        #             if yawny>=3:
        #                 sound = st.empty()
        #                 sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
        #                 time.sleep(2)  # wait for 2 seconds to finish the playing of the audio
        #                 sound.empty()  # optionally delete the element afterwards

        #         else:
        #             yawny=0

        if len(results.xywh[0])==0 and len(results_y.xywh[0])==1:

                yawn_dconf=results_y.xywh[0][0][4]
                yawn_class=results_y.xywh[0][0][5]            


                if (yawn_dconf.item()>0.65 and yawn_class.item()==0.0):
                    yawny+=1
                    if yawny>=3:
                        sound = st.empty()
                        sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                        time.sleep(2)  # wait for 2 seconds to finish the playing of the audio
                        sound.empty()  # optionally delete the element afterwards
    else:
        st.write('Stopped')


       
        
        
        
