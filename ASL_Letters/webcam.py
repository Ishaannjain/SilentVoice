import cv2
import torch
from PIL import Image
from cnn import ASLCNN, val_transform
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_asl_model.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint['classes']

model = ASLCNN(num_classes=len(classes)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Error: Could not open webcam.')
    exit()

print('Press Q to quit.')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    box_size = 300
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]
    pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    tensor = val_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    letter = classes[predicted.item()]
    conf = confidence.item() * 100

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f'{letter}  {conf:.1f}%'
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow('ASL Recognition - Press Q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
