# smart-monitoring-system
import cv2
import mysql.connector
import numpy as np
import face_recognition
import face

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained face recognition model
model = cv2.face.LBPHFaceRecognizer_create()
model.read('face_model.xml')

# Connect to the MySQL database
mydb = mysql.connector.connect(
  host="localhost",
  user="FR_Project",
  password="Adaikkammai",
  database="FR_Database"
)
mycursor = mydb.cursor()

# Create a table to store the faces and their labels
mycursor.execute("CREATE TABLE IF NOT EXISTS faces (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), face LONGBLOB)")

# Load the sample image or video stream
cap = cv2.VideoCapture(0)

# Loop over frames in the video stream
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # For each detected face, extract the face region and recognize the face
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = model.predict(face)
        
        # Draw a rectangle around the detected face and label it with the recognized name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Store the detected face and its label in the database
        encoded_face = cv2.imencode('.jpg', face)[1].tobytes()
        sql = "INSERT INTO faces (name, face) VALUES (%s, %s)"
        val = ("Person " + str(label), encoded_face)
        mycursor.execute(sql, val)
        mydb.commit()
    
    # Display the frame with the detected faces
    cv2.imshow('Face Detection and Recognition', frame)
    
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
