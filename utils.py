#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description:
-------------------------------------------------
"""

import os
import tempfile
import time

import cv2
import streamlit as st
from PIL import Image

import config
from ultralytics import YOLO

t0 = time.time()
count = 0
timeplay = 0
status = 0
last = 0
keyword = ""


def _display_detected_frames(conf, model, st_count, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_count (Streamlit object): A Streamlit object to display the object count.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video frame.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    # image = cv2.resize(image, (720, int(720 * (9 / 16))))
    # Predict the objects in the image using YOLOv8 model
    global t0, timeplay, keyword, count
    color = ''
    res = model.predict(image, conf=conf)
    result_string = res[0].verbose()
    if result_string.__contains__("GreenSafe"):
        keyword = "GreenSafe"
        color = 'green'
        check_status(1)
    elif result_string.__contains__("GreenRisk"):
        keyword = "GreenRisk"
        color = 'orange'
        check_status(2)
    elif result_string.__contains__("RedNoEntry"):
        keyword = "RedNoEntry"
        color = 'red'
        check_status(3)
    st_count.subheader("Now: " + ":" + color + "[" + keyword + "]")  # text realtime update
    autoplay_audio(keyword, 10)  # audio player
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        st.sidebar.header("Step5.Press Execution")
        if st.button("Predict"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.header("Uploaded Video")
        st.video(source_video)

    if source_video:
        st.subheader("Step5.Press Predict")
        flag_predict = st.button("Predict")
        flag_shut_down = st.button("Shut Down")
        if flag_predict:

            with st.spinner("Video Predicting..."):
                try:
                    config.OBJECT_COUNTER1 = None
                    config.OBJECT_COUNTER = None
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_count = st.empty()
                    st_frame = st.empty()
                    while not flag_shut_down and vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf, model, st_count, st_frame, image)
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Shut Down"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_count = st.empty()
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_count,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")


def autoplay_audio(kw, t):
    global t0, count, timeplay
    if count > 10:
        count = 0
        if time.time() - timeplay > t:  # audio delay
            os.system(
                'start /b ffmpeg/bin/ffplay.exe -autoexit -nodisp alarm_soundEffect/' + kw + '.mp3')  # audio player
            timeplay = time.time()  # System Clock


def check_status(c):
    global status, last, count
    if status == 0 or status == last:  # simple filter
        count += 1
    else:
        count -= 1
    last = status
    status = c
