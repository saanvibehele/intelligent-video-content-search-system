import os
import cv2
import time
from ultralytics import YOLOWorld
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pickle

model = YOLOWorld("yolov8s-world.pt")
ic_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
ic_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def detect_n_save(frame, frame_no, data):
    results = model.predict(frame)
    boxes = results[0].boxes

    class_indices = boxes.cls
    labels = [results[0].names[int(cls_idx)] for cls_idx in class_indices]

    data[frame_no] = {}
    
    for idx, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        box_ = (x1, y1, x2, y2)
        data[frame_no][labels[idx]] = box_

    raw_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = ic_processor(raw_image, return_tensors="pt")
    out = ic_model.generate(**inputs, max_new_tokens=50)
    desc = ic_processor.decode(out[0], skip_special_tokens=True)
    data[frame_no]['desc'] = desc

def save_video(rtsp_url, q):
    cap = cv2.VideoCapture(rtsp_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)
    frame_number = 0
    frame_count = 0
    video_interval = 30
    video_count = 1
    video_dir = "static/videos"
    print("url: ",rtsp_url)

    # if  os.path.isfile(video_dir + "/output_1.webm"):
    #     print("video saved")
    #     return

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    print("fourcc: ",fourcc)
    out = cv2.VideoWriter(f'{video_dir}/output_1.webm', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        if frame_count % frame_interval == 0:
            q.put((frame, frame_number))
            print(f"frame {frame_number} sent to queue ")
            frame_number += 1
            if frame_number % video_interval == 0:
                video_count += 1
                out = cv2.VideoWriter(f"{video_dir}/output_{video_count}.webm", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        frame_count += 1

    cap.release()
    out.release()

def process_each_frame(q, data):
    time.sleep(0.2)
    print("process 2 started")
    save_dir = "static/frames"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    flag = 0

    while not q.empty():
        print("entering while")
        flag = 1
        if not q.empty():
            frame, frame_number = q.get()
            print(f"frame {frame_number} being processed and saved")
            frame_path = os.path.join(save_dir, f'frame_{frame_number}.jpg')
            cv2.imwrite(frame_path, frame)
            detect_n_save(frame, f"frame{frame_number}", data)
        else:
            time.sleep(0.5)
            break
    print("process 2 end")
    if flag == 1:
        with open("data2.pkl", "wb") as file:
            pickle.dump(data, file)
