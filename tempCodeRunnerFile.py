model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')
model.conf = 0.5  # Đặt ngưỡng confidence