import cv2
import torch
from model import get_model
from torchvision.transforms import ToTensor

num_classes = 4 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes).to(device)

checkpoint_path = "models/model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval() 

CONFIDENCE_THRESHOLD = 0.5

video_capture = cv2.VideoCapture(0)  
if not video_capture.isOpened():
    print("Error: Could not open video device.")
    exit()

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = ToTensor()(frame_rgb).unsqueeze(0).to(device) 
    return frame_tensor

def draw_predictions(frame, predictions):
    boxes = predictions[0]["boxes"]
    labels = predictions[0]["labels"]
    scores = predictions[0]["scores"]

    label_map = {1: "yellow", 2: "red", 3: "blue"}

    for box, label, score in zip(boxes, labels, scores):
        if score >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

            color_name = label_map.get(label.item(), "unknown")
            label_text = f"{color_name} game piece"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 

    return frame

print("Starting video stream... Press 'q' to quit.")
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_tensor = preprocess_frame(frame)

    with torch.no_grad():
        predictions = model(frame_tensor)

    frame = draw_predictions(frame, predictions)

    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()