import cv2
import numpy as np
import tensorflow as tf

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def preprocess_frame(frame, input_size):
    resized_frame = cv2.resize(frame, input_size)
    normalized_frame = resized_frame / 255.0 
    input_data = np.expand_dims(normalized_frame, axis=0).astype(np.float32)
    return input_data

def postprocess_output(frame, boxes, classes, scores, input_size, labels, threshold=0.5):
    h, w, _ = frame.shape
    scale_x = w / input_size[0]
    scale_y = h / input_size[1]

    for i, score in enumerate(scores):
        if score > threshold:
            box = boxes[i]
            label = int(classes[i])
            ymin, xmin, ymax, xmax = box

            xmin = int(xmin * scale_x)
            ymin = int(ymin * scale_y)
            xmax = int(xmax * scale_x)
            ymax = int(ymax * scale_y)

            color = (0, 255, 0) 
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, f"{labels[label]}: {score:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def main():
    model_path = "models/model.tflite" 
    labels = {0: "red", 1: "blue", 2: "yellow"}  
    input_size = (640, 640)  
    confidence_threshold = 0.5

    print("Loading TFLite model...")
    interpreter, input_details, output_details = load_tflite_model(model_path)
    print("Model loaded successfully.")

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("Starting webcam. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        input_data = preprocess_frame(frame, input_size)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])  
        classes = interpreter.get_tensor(output_details[1]['index']) 
        scores = interpreter.get_tensor(output_details[2]['index']) 

        output_frame = postprocess_output(frame, boxes[0], classes[0], scores[0], input_size, labels, threshold=confidence_threshold)
        cv2.imshow("Object Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
