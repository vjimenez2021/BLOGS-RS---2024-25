# BLOGS-RS---2024-25
This is my repository to upload my Robótica de Servicio`s blogs.


# Rob-tica-de-Servicio-24-25

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
