import cv2

thres = 0.45  # Threshold to detect object

cap = cv2.VideoCapture(1)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height
cap.set(10, 70)   # Set brightness

# Load class names
classNames = []
classFile = 'coco.names'  # Use single quotes for consistency
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')  # Corrected line break character

# Load model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Initialize the model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    
    if not success:  # Check if the frame was captured successfully
        print("Failed to capture image")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)  # Use double quotes for consistency
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Add exit condition
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
