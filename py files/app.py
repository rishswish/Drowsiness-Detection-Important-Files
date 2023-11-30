import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
st.title('Hello')
model = None
confidence = .25


def infer_image(img, size=None):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='last_eye.pt', force_reload=True)
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


img_or_vid=st.selectbox('Input as image or Input as Video?',('Image','Video'))

if img_or_vid=='Image':
    # picture = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    picture = st.camera_input("Take a picture")
    if picture:
        pics=Image.open(picture)
        pics_array=np.array(pics)

        col1, col2 = st.columns(2)
        with col1:
            st.image(picture, caption="Selected Image")
        with col2:
            img = infer_image(pics_array)
            st.image(img, caption="Model prediction")

# if img_or_vid=='Video':
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()
    
#     # Make detections 
#         results = model(frame)
    
#         cv2.imshow('YOLO', np.squeeze(results.render()))
    
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#          break
#     cap.release()
#     cv2.destroyAllWindows() 

       
        
        
        