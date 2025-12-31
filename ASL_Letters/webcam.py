import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import os

# Same model architecture
class ASL_CNN(nn.Module):
    def __init__(self, num_classes):
        super(ASL_CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def load_model_and_classes(model_path, train_data_path, device):
    """Load model and get actual class names from training data"""
    # Get class names from the training directory structure
    if os.path.exists(train_data_path):
        dataset = datasets.ImageFolder(train_data_path)
        class_names = dataset.classes
        print(f"✓ Found {len(class_names)} classes:")
        for i, name in enumerate(class_names):
            print(f"   {i}: {name}")
    else:
        print(f"Error: Training path not found at {train_data_path}")
        return None, None
    
    num_classes = len(class_names)
    
    # Load model
    model = ASL_CNN(num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"\n✓ Model loaded successfully from {model_path}\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    return model, class_names


def preprocess_frame(frame, image_size=200):
    """Convert OpenCV frame to model input tensor"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(pil_image)
    tensor = tensor.unsqueeze(0)
    return tensor


def predict_sign(model, frame, device, class_names, image_size=200, top_k=5):
    """Predict ASL letter from frame - returns top-k predictions"""
    tensor = preprocess_frame(frame, image_size).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(class_names)))
        
        predictions = []
        for i in range(len(top_indices[0])):
            pred_class = class_names[top_indices[0][i].item()]
            confidence = top_probs[0][i].item() * 100
            predictions.append((pred_class, confidence))
    
    return predictions


def draw_roi_box(frame):
    """Draw a region of interest box where hand should be placed"""
    h, w = frame.shape[:2]
    # Center box
    box_size = 300
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    
    # Draw semi-transparent box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw corner markers
    marker_len = 30
    thickness = 3
    color = (0, 255, 0)
    
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + marker_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + marker_len), color, thickness)
    
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - marker_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + marker_len), color, thickness)
    
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + marker_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - marker_len), color, thickness)
    
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - marker_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - marker_len), color, thickness)
    
    # Instruction text
    cv2.putText(frame, "Place hand in this box", 
                (x1 + 40, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)


def main():
    # Configuration
    MODEL_PATH = "models/best_asl_cnn.pt"
    TRAIN_DATA_PATH = "data/Datasets/asl_alphabet_train"
    IMAGE_SIZE = 200
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train the model first.")
        return
    
    # Load model and get actual class names
    model, class_names = load_model_and_classes(MODEL_PATH, TRAIN_DATA_PATH, device)
    
    if model is None or class_names is None:
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("="*70)
    print("ASL Letter Recognition - Real Time")
    print("="*70)
    print("\n⚠️  IMPORTANT: Your model was trained on only ~7 images per letter!")
    print("   Performance may be limited. Consider retraining with more data.\n")
    print("Instructions:")
    print("  • Position your hand clearly in the green box")
    print("  • Use good lighting and a plain background")
    print("  • Hold steady for better predictions")
    print("\nControls:")
    print("  q - Quit")
    print("  s - Save current frame with prediction")
    print("  d - Toggle debug info (top-5 predictions)")
    print("  h - Toggle help overlay")
    print("\nStarting webcam...\n")
    
    frame_count = 0
    show_debug = True
    show_help = True
    predictions = [("?", 0.0)] * 5
    
    # For smoothing predictions
    prediction_history = []
    history_size = 5
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Draw ROI box
        if show_help:
            draw_roi_box(frame)
        
        # Get prediction every 2 frames for better responsiveness
        if frame_count % 2 == 0:
            current_predictions = predict_sign(model, frame, device, class_names, IMAGE_SIZE, top_k=5)
            predictions = current_predictions
            
            # Add to history for smoothing
            if predictions[0][1] > 30:  # Only add if confidence > 30%
                prediction_history.append(predictions[0][0])
                if len(prediction_history) > history_size:
                    prediction_history.pop(0)
        
        # Get most common prediction from history (smoothed prediction)
        if prediction_history:
            from collections import Counter
            smoothed_letter = Counter(prediction_history).most_common(1)[0][0]
        else:
            smoothed_letter = predictions[0][0]
        
        # Main prediction panel
        pred_letter, pred_conf = predictions[0]
        panel_height = 140 if show_debug else 100
        
        cv2.rectangle(frame, (10, 10), (min(500, w-10), panel_height), (0, 0, 0), -1)
        
        # Current prediction
        cv2.putText(frame, f"Predicting: {pred_letter}", 
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 255, 0), 3)
        
        # Confidence with color coding
        if pred_conf > 70:
            color, status = (0, 255, 0), "HIGH"
        elif pred_conf > 40:
            color, status = (0, 165, 255), "MEDIUM"
        else:
            color, status = (0, 0, 255), "LOW"
            
        cv2.putText(frame, f"Confidence: {pred_conf:.1f}% ({status})", 
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)
        
        # Smoothed prediction
        if smoothed_letter != pred_letter:
            cv2.putText(frame, f"Stable: {smoothed_letter}", 
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (100, 200, 255), 1)
        
        # Show top-5 predictions in debug mode
        if show_debug:
            cv2.rectangle(frame, (10, panel_height + 10), (min(400, w-10), panel_height + 170), (40, 40, 40), -1)
            cv2.putText(frame, "Top 5 Predictions:", 
                       (20, panel_height + 35), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            for i, (letter, conf) in enumerate(predictions[:5]):
                y_pos = panel_height + 60 + i * 22
                bar_width = int(conf * 2.5)  # Scale for visualization
                
                # Draw confidence bar
                cv2.rectangle(frame, (180, y_pos - 12), (180 + bar_width, y_pos - 2), 
                            (0, 255, 0) if i == 0 else (100, 100, 100), -1)
                
                cv2.putText(frame, f"{i+1}. {letter:>7}: {conf:5.1f}%", 
                           (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.45, (200, 200, 200), 1)
        
        # Instructions at bottom
        instructions = "q:Quit | s:Save | d:Debug | h:Help"
        cv2.putText(frame, instructions, 
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        
        # Warning if confidence is very low
        if pred_conf < 20:
            cv2.putText(frame, "⚠ Uncertain - adjust lighting/position", 
                       (w//2 - 200, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 255), 1)
        
        # Display frame
        cv2.imshow('ASL Letter Recognition', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            filename = f"captured_{pred_letter}_{smoothed_letter}_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved: {filename}")
            print(f"  Instant: {pred_letter} ({pred_conf:.1f}%)")
            print(f"  Stable:  {smoothed_letter}")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
        elif key == ord('h'):
            show_help = not show_help
            print(f"Help overlay: {'ON' if show_help else 'OFF'}")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")
    print("Goodbye!")


if __name__ == "__main__":
    main()