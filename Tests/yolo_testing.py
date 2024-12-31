from ultralytics import YOLO

#model = YOLO('yolov8m')
model = YOLO('Finetuned Yolo/best(1).pt')

#first_test = model.predict('First_test/Video/08fd33_4.mp4', save = True)
second_test = model.predict('Tests/Video/08fd33_4.mp4', save = True)