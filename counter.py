import cv2
import numpy as np

#ENTER OUR ESP IP
ip_url = "http://192.168.254.141:81/stream"
cap = cv2.VideoCapture(ip_url)

#CREAT BGS
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

#OUR COUNTERS
metal_count = 0       #LEFT TO RIGHT
non_metal_count = 0   #RIGHT TO LEFT
crossed_objects = {}  #STORE OBJECT

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to fetch frame. Check ESP32-CAM connection.")
        break

    #FRAME CHE DIMENSIONS
    height, width, _ = frame.shape

    #TWO VERTICAL REFERENCE LINE
    left_line = int(width * 0.3)   #30 PER FROM LEFT
    right_line = int(width * 0.7)  #70 PER FROM LEFT

    #BGS USE KELA
    fg_mask = bg_subtractor.apply(frame)

    #DETECTIONS PART
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #PROCESS OF DETECTION
    for contour in contours:
        if cv2.contourArea(contour) > 500:  #IGNORE SMALL NOICE OBJECT
            x, y, w, h = cv2.boundingRect(contour)
            obj_center = x + w // 2  #CENTER

            #TRACK
            object_id = id(contour)  #UNIQUE ID OF OBJECT

            if object_id not in crossed_objects:
                crossed_objects[object_id] = obj_center  #STORE KELI INITIAL

            prev_position = crossed_objects[object_id]
            crossed_objects[object_id] = obj_center  #UPDATE KELI NEW INFO

            #METAL IF LEFT TO RIGHT
            if prev_position < left_line and obj_center > right_line:
                metal_count += 1
                print(f"Metal Count Increased: {metal_count}")
                del crossed_objects[object_id]  #REMOVE TRACKING

            #NON METAL IF RIGHT TO LEFT
            elif prev_position > right_line and obj_center < left_line:
                non_metal_count += 1
                print(f"Non-Metal Count Increased: {non_metal_count}")
                del crossed_objects[object_id]  #REMOVE TRACKING

            #OBJECT BOUNDING BOX
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #DRAWING VERTICAL LINE
    cv2.line(frame, (left_line, 0), (left_line, height), (0, 255, 0), 2)
    cv2.line(frame, (right_line, 0), (right_line, height), (0, 255, 0), 2)

    #DISPLAYING COUNT
    font_scale = 0.6  #SMALL TEXT
    font_thickness = 2
    text_color = (0, 255, 255)  #CSK COLOR

    cv2.putText(frame, f"Metal: {metal_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    cv2.putText(frame, f"Non-Metal: {non_metal_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    #SHOW OP FRAME
    cv2.imshow("ESP32-CAM Object Counting", frame)

    #PRESS Q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#RELEASING LABOURERS 
cap.release()
cv2.destroyAllWindows()
