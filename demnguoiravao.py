import cv2
import torch
from datetime import datetime
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import pygame
import threading

pygame.mixer.init()

def play_warning_sound():
    try:
        pygame.mixer.music.load("warning.mp3")
        pygame.mixer.music.play()
    except Exception as e:
        print("Lỗi phát âm thanh:", e)

# Load model đã train xong
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')

# Đếm số người ra/vào theo chiều ngang
entry_line = 250  # Line giữa khung hình theo chiều ngang
exit_line = 400
in_count = 0
out_count = 0

track_history = {}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không thể mở nguồn video")
    exit()

def update_frame():
    global in_count, out_count, track_history

    ret, frame = cap.read()
    if not ret:
        return

    results = model(frame)
    detections = results.xyxy[0]
    current_centroids = []

    for i, (*box, conf, cls) in enumerate(detections):
        if int(cls) != 0:  # Chỉ theo dõi người
            continue

        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        current_centroids.append((cx, cy))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

    for idx, (cx, cy) in enumerate(current_centroids):
        if idx in track_history:
            prev_cx, prev_cy = track_history[idx]
            if prev_cx < entry_line and cx >= entry_line:
                in_count += 1
            elif prev_cx > exit_line and cx <= exit_line:
                out_count += 1

        track_history[idx] = (cx, cy)

    # Vẽ line theo chiều dọc
    cv2.line(frame, (entry_line, 0), (entry_line, frame.shape[0]), (255, 0, 0), 2)
    cv2.line(frame, (exit_line, 0), (exit_line, frame.shape[0]), (0, 0, 255), 2)

    # Hiển thị số lượng
    cv2.putText(frame, f"In: {in_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Out: {out_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

    people_inside = in_count - out_count
    current_time = datetime.now().strftime("%H:%M:%S")
    if people_inside > 0:
        cv2.putText(frame, "WARNING: People still inside!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        print(f"[{current_time}] Warning! Có {people_inside} người vẫn còn trong xe.")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    lbl.imgtk = imgtk
    lbl.configure(image=imgtk)
    lbl.after(10, update_frame)

def check_and_warn():
    people_inside = in_count - out_count
    if people_inside > 0:
        threading.Thread(target=play_warning_sound).start()

root = Tk()
root.title("Đếm người ra vào")

lbl = Label(root)
lbl.pack()

btn = Button(root, text="Kiểm tra cảnh báo", font=("Arial", 14), bg="red", fg="white", command=check_and_warn)
btn.pack(pady=10)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
