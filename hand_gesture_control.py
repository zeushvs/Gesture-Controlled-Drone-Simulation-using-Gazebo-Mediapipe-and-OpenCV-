# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from geopy.distance import geodesic


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Connect to the vehicle
print('Connecting...')
vehicle = connect('udp:127.0.0.1:14551')

# Setup the commanded flying speed
gnd_speed = 5  # [m/s]

# Define arm and takeoff
def arm_and_takeoff(altitude):
    while not vehicle.is_armable:
        print("waiting to be armable")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        time.sleep(1)

    print("Taking Off")
    vehicle.simple_takeoff(altitude)

    while True:
        v_alt = vehicle.location.global_relative_frame.alt
        print(">> Altitude = %.1f m" % v_alt)
        if v_alt >= altitude - 1.0:
            print("Target altitude reached")
            break
        time.sleep(1)

# Define the function for sending mavlink velocity command in body frame
def set_velocity_body(vehicle, vx, vy, vz):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,  #-- BITMASK -> Consider only the velocities
        0, 0, 0,            #-- POSITION
        vx, vy, vz,         #-- VELOCITY
        0, 0, 0,            #-- ACCELERATIONS
        0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

# Define the hand gesture event function
def hand_gesture_event(gesture):
    # Modify this function to control the drone based on the recognized gestures
    if gesture == 'thumbs up' or gesture == 'smile':
        set_velocity_body(vehicle,0, 0,-gnd_speed)
    elif gesture == 'thumbs down':
        set_velocity_body(vehicle,0 ,0,gnd_speed)
    elif gesture == 'peace' or gesture == 'okay':
        set_velocity_body(vehicle, 0 , gnd_speed, 0 )
    elif gesture == 'call me':
        set_velocity_body(vehicle,0, -gnd_speed,0)
    elif gesture == 'stop' or gesture == 'live long':
        set_velocity_body(vehicle,  -gnd_speed, 0 ,0 ) # Adjust for your drone's altitude control
    elif gesture == 'fist' or gesture == 'rock':
        set_velocity_body(vehicle,  gnd_speed , 0 ,0)  # Adjust for your drone's altitude control

# Main function
def main():
    # Takeoff
    arm_and_takeoff(10)

    # Read the webcam and perform hand gesture recognition
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        x, y, c = frame.shape

        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)
        print("Hand Landmarks Result:", result)

        className = ''

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                prediction = model.predict([landmarks])
                print("Prediction:", prediction)
                classID = np.argmax(prediction)
                print("Class ID:", classID)

                if 0 <= classID < len(classNames):
                    className = classNames[classID]
                else:
                    className = "Unknown"

        if className:
            hand_gesture_event(className)

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Output", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
