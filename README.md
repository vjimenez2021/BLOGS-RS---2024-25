# BLOGS-RS---2024-25
This is my repository to upload my Robótica de Servicio`s blogs.

Index:
- [Go to P1 - Localized Vaccum Cleaner](#p1-localized-vaccum-cleaner)
- [Go to P2 - Rescue Drone](#p2-rescue-drone)
- [Go to P3 - Autoparking](#p3-autoparking)
- [Go to P4 - Warehouse](#p4-warehouse)

## P1-Localized Vaccum Cleaner

At first, I realized that the original map had the dimensions of (1012, 1013) pixels. Because of this, the first thing I did, was to create an empty matrix with the same dimensions to recreate the original map and start eroding it and redimensioning it.

```python
# Get the map image
map_image = GUI.getMap('/RoboticsAcademy/exercises/static/exercises/vacuum_cleaner_loc_newmanager/resources/images/mapgrannyannie.png')

# Creating new matrix to make the new map and coloring it
new_map = np.zeros((1012, 1013), dtype=np.uint8)
for i in range(map_image.shape[0]):
    for j in range(map_image.shape[1]):
        if np.array_equal(map_image[i, j], [255, 255, 255]):
            new_map[i, j] = 127
        else:
            new_map[i, j] = 0
```

Once I had the matrix, I filled it up with the colors of the pixels in the map so if I encountered a black pixel or a white one I would fill the matrix up with that colour.
This made me had a 1012x1013 matrix with the map and then I could erode it to make it 720x720 so I could make cells of 20x20 pixels or 15x15 pixels or 16x16 pixels and had nice dimensions for the cell.

After I got the matrix I started with the conversion system:

First I measured the map and I calculated 9.75 meters from bottom to top and 9.75 meters from left to right in the whole map. With this measurements I could make a simple divission to know the scale factor which is 720 pixel divided by 9.75 meters makes a total of 75 pixels each meter more or less.

Then with the conversion matrix to get the pixel knowing the coordinates in the real world (from world to pixel) I could represent my robot in the new map image.

![image](https://github.com/user-attachments/assets/386ec131-0c80-46bc-9db0-e254603dc6bc)

![Screenshot from 2024-10-10 11-12-20](https://github.com/user-attachments/assets/858143da-29ff-4c8d-868c-4fda11190092)


Thanks to this, I knew in every moment in which pixel my robot was:

![image](https://github.com/user-attachments/assets/08bdcbf9-09ea-4f56-b934-0039debfe08e)


After knowing this, it was the time to classify all my cells into three diferent groups:

 - Cells to avoid: This group is made out of cells which contains at least one black pixel. This is made like this to avoid going near obstacles as the robot movement and localization is not 100% precise.ç
 - Cells to clean: This group is made out of all the cells made exclusively by white pixels so I knew which cell I needed to go through in order to clean the whole house.
 - Cells cleaned: Once the robot went through a cell to clean, this cell was removed from that group and was included in this group so the robot didnt go through the same cell twice if it was not needed.

Now we needed the algorithm to classify every cell:

```python
# Function to classify the cells depending on whether they contain black pixels or not.
def classify_cells():
    global cells_to_avoid, cells_to_clean

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Obtain the pixels of the current cell (20x20 pixels)
            cell_pixels = map_image_resized[row * grid_size:(row + 1) * grid_size, col * grid_size:(col + 1) * grid_size]
            
            # If at least one pixel is black (obstacle), add it to cells_to_avoid
            if np.any(cell_pixels == 0):
                cells_to_avoid.append((row, col))
            else:
                # If all pixels are white, add to cells_to_clean
                cells_to_clean.append((row, col))
```

At this point I already knew to which group belonged each cell. Now I only needed an algorithm that calculated the path that the robot would follow with a priority of movements.
In my case, the movement priority is:

    1.- Left
    2.- Down
    3.- Right
    4.- Up

I say "Up" and "Down" taking into account that the map is a 2D map so there are only four possible movements.

In order to move my robot, I had to create another funciton to convert from 2d to 3d. I needed to use this function to convert from a pixel to world coordinates.

I made it like this as I had the following idea:

To move my robot first I would search the target cell and extract a pixel from that cell to hace the target pixel. Then, I would convert that pixel to world 3D coordinates. Once I have all of this, I would align my robot to the new target 
coordinates and once it is aligned, I would move my robot in a straight line in order to reach the new cell.

Furthermore, I needed a backtracking algorithm so I decided to create my interest points array. In this array I stored all the points where I had more than one movement possibility and if in some point
I didnt have anymore movements I would just return to my interest points to continue.
But this was had an error, I was storing every interest point but I didnt remove them when they werent useful anymore. So, after realizing I had this error, I decided to remove every interest point if its adjacent cells were already visited by the robot.

```python
# Function to know if a cell has neighbours clean or not
def has_clean_neighbors(cell):
    row, col = cell
    neighbors = [(row, col - 1), (row + 1, col), (row, col + 1), (row - 1, col)]
    for neighbor in neighbors:
        if neighbor in cells_to_clean:
            return True  # If any neighbours is not cleaned, True
    return False  # If all the neighbours are cleaned, False
```

If my interest point had any neighbour which wasnt cleaned, it was useful for me to store it so I could return to them to visit non-visited areas. But, if all the neighbours were already cleaned, it had no sense to return back to that interest point.

Also, I made another algorithm to return back to my interest point as at first, I made all the way back until I reached my interest point but this took me a long time. I decided to use BFS (Bread First Search) so I explored all the neighbours of the current cell and expanded them until I reached the interest point I needed to go. Once I found the interest point, I stored the path to it and the robot followed the path until it got to the interest point to continue with the cleaning path.

After making this algorithm I needed to adjust some of the constants and the grid size (cell size) to clean the most area possible and not having any problems with the obstacles.

If I made a very small cell size, my robot cleaned more surface of the house but some of the obstacles as the tables legs or the chairs were a big problem:

![Screenshot from 2024-10-07 18-17-23](https://github.com/user-attachments/assets/ba9fcb59-21d3-4fdf-a1a7-7b5c1e296937)

And if I made the cell size a little big bigger, the robot just covered a little area.

After all of this problems I decided to use another OpenCV technique: Dilate the black pixels in order to create "imaginary bigger objects" so I could have a security distance from the real obstacle.
After some tries and adjusting a lot the different constants I achieved to clean the whole house (despite the surface has some areas where the robot doesnt go).

Here is an image of how the robot cleaned the house:

![Screenshot from 2024-10-07 22-28-55](https://github.com/user-attachments/assets/730cf564-a2d2-4047-afad-f73ab92eb922)

Here is the video:

https://github.com/user-attachments/assets/f03d9aed-9f15-450a-9eba-056c3115591f

This video shows how my robot cleans with a map size of 720x720 pixels which is not really good.

After this and after some tries I realized that the real problem was the size of the map as if I have a small sized map, the details are not taken into accound really much. Furtheremore, with the dilate funcion of OpenCv, the obstacles are
not represented very well and precisely.

Because of this, I decided to redimension the map into a 1000x1000 pixels size so I could get the details and precission of the obstacles better.

The new video with the resized map 1000x1000:



https://github.com/user-attachments/assets/05a27121-ffbf-498c-b36d-3b3ba93c38bd


Because of this, I have decided to maintain this configuration for my map as the area cleaned is bigger and better than the previous one.


### Other problems faced

I have faced some other problems with this exercise:
The main problem I have faced is that I have advanced in the exercise with some errors in the conversion matrix and the conversion system which I have carried until I corrected them so I could do it better.
Due to this and to know where the problem was I have also created some functions to depure my code. For example to know where the robot is (know in which cell it is):

```python
# Check on what cell the robot is
def print_robot_cell():
    pos_2d = transform_3d_to_2d()

    x, y = pos_2d
    print(f"Robot position in pixels: x = {x}, y = {y}")

    cell_x, cell_y = get_robot_cell(pos_2d)
    print(f"Robot is in cell: ({cell_x}, {cell_y})")

    if (cell_x, cell_y) in cells_to_clean:
        cells_to_clean.remove((cell_x, cell_y))
        cells_cleaned.append((cell_x, cell_y))

# Function to know the group a cell belongs to.
def check_cell_status(row, col):
    if (row, col) in cells_to_clean:
        print(f"Cell ({col}, {row}) is in 'cells_to_clean'.")
    elif (row, col) in cells_cleaned:
        print(f"Cell ({col}, {row}) is in 'cells_cleaned'.")
    elif (row, col) in cells_to_avoid:
        print(f"Cell ({col}, {row}) is in 'cells_to_avoid'.")
    else:
        print(f"Cell ({col}, {row}) is not classified.")
```

With this two functions I could know the group a cell belonged to and know if I was classifying them in the correct way or not.
I could also know in which cell was the robot detecting it was and know if I had the autolocalization in the correct way or not.


## P2-Rescue Drone

To cover the goal of this exercise, first I decided to learn how to move the drone as we had three different options:

```python
# 1. Position control

    HAL.set_cmd_pos(x, y, z, az) # Commands the position (x,y,z) of the drone, in m and the yaw angle (az) (in rad) taking as reference the first takeoff point (map frame)

# 2. Velocity control

    HAL.set_cmd_vel(vx, vy, vz, az) # Commands the linear velocity of the drone in the x, y and z directions (in m/s) and the yaw rate (az) (rad/s) in its body fixed frame

# 3. Mixed control

    HAL.set_cmd_mix(vx, vy, z, az) # Commands the linear velocity of the drone in the x, y directions (in m/s), the height (z) related to the takeoff point and the yaw rate (az) (in rad/s)

```

I tried the three of them bust the best option was the first one: HAL.set_cmd_pos().

This is because in order to cover the impact area I wanted to cover it by using a square area and going from side to side of the square begining at the bottom of it and finishing at it's top.

By using this coverage method, I had made an "imaginary" square which center is the area where the impact zone is known to be ("While survivors are known to be close to 40º16’47.23” N, 3º49’01.78” W").
So, by knowing this, I have built a square to cover the area and its surroundings in search of survivors.

For the searching, I used the face detection. I used this python code for face detecting with the image of my Ventral Camera.

```python
from __future__ import print_function
import cv2 as cv
import argparse
def detectAndDisplay(frame):
 frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 frame_gray = cv.equalizeHist(frame_gray)
 #-- Detect faces
 faces = face_cascade.detectMultiScale(frame_gray)
 for (x,y,w,h) in faces:
 center = (x + w//2, y + h//2)
 frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
 faceROI = frame_gray[y:y+h,x:x+w]
 #-- In each face, detect eyes
 eyes = eyes_cascade.detectMultiScale(faceROI)
 for (x2,y2,w2,h2) in eyes:
 eye_center = (x + x2 + w2//2, y + y2 + h2//2)
 radius = int(round((w2 + h2)*0.25))
 frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
 cv.imshow('Capture - Face detection', frame)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
 print('--(!)Error loading face cascade')
 exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
 print('--(!)Error loading eyes cascade')
 exit(0)
camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
 print('--(!)Error opening video capture')
 exit(0)
while True:
 ret, frame = cap.read()
 if frame is None:
 print('--(!) No captured frame -- Break!')
 break
 detectAndDisplay(frame)
 if cv.waitKey(10) == 27:
 break
```

After this, my drone would detect every survivor facing frontally towards the ventral camera. This made the drone to detect some survivors but not all of them as if the survivor was not facing correctly towards the drone, it wouldn´t detect it.
In order to correct this, I have implemented a function to constantly rotate the image captured by the ventral camera in order to make all the survivors appear frontally so I could detect them with the detection algorithm.

Also, in order to carry out the battery level, I have implemented a simple counter that starts in a random numbre between 17500 and 22500 and if it reaches the 15% of the original quantity, the drone will go back to the boat.

### Video of the demonstration of the battery



https://github.com/user-attachments/assets/41b94d9e-c5e3-4944-92d7-2b6af9c491d9


In this video I have set a very low battery to show how the drone goes back if the battery is low.
As you can see, when the drone detects that the battery is low, it goes back to the boat and makes the landing.


### Various problems
I have encountered various problems such in the face detections because the distance between the registered positions of the survivors was not enough so it detected more bodies than the 6 they were.
Adjusting the distances, and some other variables such the step size (which were also adjusted) it detected the survivors correctly.
Also at the time of the landing, the drone didnt go exactly to (0,0) so I needed to be constantly calculating the distance to the boat so when it was perfectly centered it would land at the top of the boat.

With the battery I also had to do a bunch of tries in order to know the range of battery I needed to let the drone finish covering the area. If I set the battery to low, the drone didnt get to the zone correctly
but if I set the battery to high, I couldnt demonstrate that it worked. So because of this I have made two videos to demonstrate that it works perfectly.


### Video of the functioning

https://github.com/user-attachments/assets/863309e4-b424-4289-81cd-b585de67b5b8

## P3-Autoparking

This project aims to develop an autonomous parking system capable of parallel parking a simulated vehicle between two parked cars or next to a single parked car. The vehicle is controlled by a series of maneuvers based on laser sensor data, orientation adjustments, and precise movement control. Throughout the development of this system, I encountered several challenges that required refining the approach and logic multiple times. Here, I’ll explain the steps I took, the difficulties I faced, and the solutions I implemented.

### Overview of the problem

The goal of this project was to simulate an autonomous parallel parking system. The system uses sensor data to detect the environment, align the vehicle parallel to parked cars, and execute a series of maneuvers to fit the vehicle into a parking space. I utilized HAL’s simulated sensors and actuators, specifically focusing on laser-based distance sensors and yaw readings, to manage the vehicle’s movements and orientation.

### Steps and code explanation

#### Aligning the Car with Parked Vehicles
The first step in the parking process is to align the car parallel to the parked vehicles. This involves calculating the slope (or "pendiente") of a line that represents the row of parked cars. I used laser sensors to detect nearby objects and gather distance data to estimate this slope. This process required adjusting the vehicle's yaw (orientation) until it became approximately parallel to the line of parked cars.

##### Difficulties Faced
One of the main challenges here was accurately calculating the slope. Initially, I found that small inaccuracies in the laser data caused the car to appear slightly tilted, even when aligned. I adjusted the tolerance for the slope and experimented with different thresholds to achieve a more consistent alignment.

#### Detecting a Suitable Parking Spot

Once the vehicle is aligned, the next step is to detect an available parking spot. The car moves forward along the row of parked cars until it detects a large enough gap that meets the threshold.

In order to enter in a good way into the spot, after detecting it, I move the vehicle a little bit forward in order to start cenering it and turning it towards the spot.

##### Difficulties Faced

Initially, the car sometimes failed to detect an appropriate gap or would detect false positives. Fine-tuning the distance parameters helped reduce these errors. I also found that occasional minor adjustments to the vehicle's orientation during this phase improved the detection accuracy.

#### Parking Maneuver Phases

The parking maneuver is divided into several phases to simplify the process and ensure precise control over the vehicle. Here’s a breakdown of each phase:

    Angle Adjustment - The car reverses at an angle to position itself for parallel parking.
    Straight Reverse - The car moves backward in a straight line until it detects the rear boundary.
    Alignment - If the car is misaligned, it performs final adjustments by moving forward and backward to achieve parallel positioning.

#### Handling Edge Cases

One unique aspect of this project was handling the scenario where only one car is parked, either in front or behind the vehicle. In such cases, the car should be able to park itself without using the usual two-car boundary. I implemented an additional check to determine if only a single parked car is present and adjusted the maneuver logic accordingly.

## Videos
### Video of Parking Between Two Cars



https://github.com/user-attachments/assets/f6f42833-3b1a-48a4-8aa7-ff8c219621e7



### Video of Parking with Only One Car in Front


https://github.com/user-attachments/assets/9a61e8c8-bab9-4330-a25c-3530bf1566b3



### Video of Parking with Only One Car in Back

https://github.com/user-attachments/assets/5afa9b0d-500c-4eb7-9814-090ac084d6c9


## P4-Warehouse
