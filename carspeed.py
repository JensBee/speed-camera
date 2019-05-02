# CarSpeed Version 2.0
"""
Script to capture moving car speed

Usage:
    carspeed.py [headless] [--config=<model> --use-webcam]

Options:
    -h --help     Show this screen.
"""

"""
Updates:
by Doug LaRue
	original project location: https://gregtinkers.wordpress.com/2016/03/25/car-speed-detector/
	Added --use-webcam option which adds USB WebCam capabilities
	Added headless commandline option so when along with the --config option the software can run headless as a systemd service
	Added --config monitoring_area.csv option
	Added automatic saving of the monitorying area to CSV file
"""

# import the necessary packages
import time
import math
import datetime
import cv2
from docopt import docopt
from pathlib import Path


class CarSpeed():
    def __init__(self, camera=None,  resolution=[640, 480],  freq=75, zero=1000):
        self.camera = camera
        self.resolution = resolution
        self.freq = freq
        if self.camera != None:
            self.camera.set_mode()
        self.config_file = None
        self.headless = False
        self.use_webcam = False
        
    def initialize(self):
        print("We are in CarSpeed Initialize function!")
        # get commandline options
        args = docopt(__doc__)
        print(args)
        #cfg = dk.load_config()
        if args['headless']:
            self.headless=True
        if args['--config']:
            config = args['--config']
            print('WITH CONFIG: self.config_file='+ config)
            self.config_file = Path(config)
            if self.config_file.is_file():
                self.read_csv()
            else:
                print("config file does NOT exist")
                self.config_file=None
        if args['--use-webcam']:
            self.use_webcam=True
 
    def setup_webcam(self):
        #global use_webcam
        if self.use_webcam:
            #import pygame
            import pygame
            import pygame.camera
            self.cam = 0
            self.resolution = (image_w, image_h)
            pygame.init()
            pygame.camera.init()
            camList = pygame.camera.list_cameras()
            print('cameras', camList)
            self.cam = pygame.camera.Camera(camList[self.cam], self.resolution, "RGB")
            self.cam.start()
            #framerate = framerate
            print('WebcamVideoStream loaded.. .warming camera')
            time.sleep(2)
        else:
            # initialize the camera. Adjust vflip and hflip to reflect your camera's orientation
            camera = PiCamera()
            camera.resolution = RESOLUTION
            camera.framerate = FPS
            camera.vflip = False
            camera.hflip = False
            rawCapture = PiRGBArray(camera, size=camera.resolution)
            # allow the camera to warm up
            time.sleep(0.9)
            
    def set_resolution(self, horiz,  vert):
        self.resolution=[horiz, vert]
        self.camera.resolution(self.resolution)

    def set_track_area(self, ix,  iy, fx,  fy):
        if fx > ix:
            self.upper_left_x = ix
            self.lower_right_x = fx
        else:
            self.upper_left_x = fx
            self.lower_right_x = ix

        if fy > iy:
            self.upper_left_y = iy
            self.lower_right_y = fy
        else:
            self.upper_left_y = fy
            self.lower_right_y = iy
        self.monitored_width = self.lower_right_x - self.upper_left_x
        self.monitored_height = self.lower_right_y - self.upper_left_y
        #TODO
        #save tracking area to file

    def get_track_area(self):
        #TODO
        #load tracking area from file or get it from the GUI
        self.set_track_area(ix,  iy, fx,  fy)
  
    def save_csv(self):
        if SAVE_CSV:
            csvfileout = "carspeed_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
            record_speed('Date,Day,Time,Speed,Image')
            csvmonfileout = "monitoringarea{}.csv".format(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
            record_mon_area('UpperLeftX,UpperLeftY,LowerRightX,LowerRightY,MonitoredWidth,MonitoredHeight')

        if self.SAVE_CSV:
            csvmonfileout = "monitoringarea{}.csv".format(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
            record_mon_area(("%d" % upper_left_x)+','+("%d" % upper_left_y)+','+("%d" % lower_right_x)+','+("%d" % lower_right_y)+','+\
                ("%d" % monitored_width)+','+("%d" % monitored_height))

    def read_csv(self):
        import csv
        #self.config_file = config_file
        with open(self.config_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    #print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    #set_track_area(self, ix,  iy, fx,  fy)
                    #print(f'\t iX {row[0]} iY {row[1]} fX {row[2]} fY {row[3]} MonitoredWidth {row[4]} MonitoredHeight {row[5]}.')
                    self.set_track_area(int(row[0]),  int(row[1]), int(row[2]),  int(row[3]))
                    line_count += 1
            #print(f'Processed {line_count} lines.')
    
    def __del__(self):
        self.camera.stop()

#<--- END OF CarSpeed class -->


#print("new Carspeed")
#car = CarSpeed()
#car.initialize()
#   #car.setup_webcam()

# place a prompt on the displayed image
def prompt_on_image(txt):
    global image
    cv2.putText(image, txt, (10, 35),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
     
# calculate speed from pixels and time
def get_speed(pixels, ftperpixel, secs):
    if secs > 0.0:
        return ((pixels * ftperpixel)/ secs) * 0.681818  
    else:
        return 0.0
 
# calculate elapsed seconds
def secs_diff(endTime, begTime):
    diff = (endTime - begTime).total_seconds()
    return diff

# record speed in .csv format
def record_speed(res):
    global csvfileout
    f = open(csvfileout, 'a')
    f.write(res+"\n")
    f.close

# record monitoring area in .csv format
def record_mon_area(res):
    global csvmonfileout
    f = open(csvmonfileout, 'a')
    f.write(res+"\n")
    f.close

# mouse callback function for drawing capture area
def draw_rectangle(event,x,y,flags,param):
    global ix,iy,fx,fy,drawing,setup_complete,image, org_image, prompt
 
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
 
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            image = org_image.copy()
            prompt_on_image(prompt)
            cv2.rectangle(image,(ix,iy),(x,y),(0,255,0),2)
  
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx,fy = x,y
        image = org_image.copy()
        prompt_on_image(prompt)
        cv2.rectangle(image,(ix,iy),(fx,fy),(0,255,0),2)
        
# define some constants
DISTANCE = 55 #<---- 54' 4.5" from center of road to outside wall.  #<---- enter your distance-to-road value here
MIN_SPEED = 0  #<---- enter the minimum speed for saving images
SAVE_CSV = True # False  #<---- record the results in .csv format in carspeed_(date).csv

THRESHOLD = 15
MIN_AREA = 175
BLURSIZE = (15,15)
IMAGEWIDTH = 640
IMAGEHEIGHT = 480
RESOLUTION = [IMAGEWIDTH,IMAGEHEIGHT]
FOV = 53.5    #<---- Field of view
FPS = 30
SHOW_BOUNDS = True
SHOW_IMAGE = True

# the following enumerated values are used to make the program more readable
WAITING = 0
TRACKING = 1
SAVING = 2
UNKNOWN = 0
LEFT_TO_RIGHT = 1
RIGHT_TO_LEFT = 2

# calculate the the width of the image at the distance specified
frame_width_ft = 2*(math.tan(math.radians(FOV*0.5))*DISTANCE)
ftperpixel = frame_width_ft / float(IMAGEWIDTH)
print("Image width in feet {} at {} from camera".format("%.0f" % frame_width_ft,"%.0f" % DISTANCE))

# state maintains the state of the speed computation process
# if starts as WAITING
# the first motion detected sets it to TRACKING
 
# if it is tracking and no motion is found or the x value moves
# out of bounds, state is set to SAVING and the speed of the object
# is calculated
# initial_x holds the x value when motion was first detected
# last_x holds the last x value before tracking was was halted
# depending upon the direction of travel, the front of the
# vehicle is either at x, or at x+w 
# (tracking_end_time - tracking_start_time) is the elapsed time
# from these the speed is calculated and displayed 
 
state = WAITING
direction = UNKNOWN
initial_x = 0
last_x = 0
 
#-- other values used in program
base_image = None
abs_chg = 0
mph = 0
secs = 0.0
ix,iy = -1,-1
fx,fy = -1,-1
drawing = False
setup_complete = False
tracking = False
text_on_image = 'No cars'
prompt = ''

headless = False
config_file = None
use_webcam = False

def read_csv():
        global config_file
        import csv
        with open(str(config_file)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    #print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    #set_track_area(self, ix,  iy, fx,  fy)
                    #print(f'\t iX {row[0]} iY {row[1]} fX {row[2]} fY {row[3]} MonitoredWidth {row[4]} MonitoredHeight {row[5]}.')
                    set_track_area(int(row[0]),  int(row[1]), int(row[2]),  int(row[3]))
                    line_count += 1
            #print(f'Processed {line_count} lines.')

def initialize():
    global headless,  config_file,  SHOW_IMAGE,  use_webcam
    print("We are in Initialize function!")
    # get commandline options
    args = docopt(__doc__)
    print(args)
    if args['headless']:
        headless=True
        SHOW_IMAGE=False
    if args['--config']:
        config = args['--config']
        print('WITH CONFIG: config_file='+ config)
        config_file = Path(config)
        if config_file.is_file():
            read_csv()
        else:
            print("config file does NOT exist")
            config_file=None
    if args['--use-webcam']:
        use_webcam=True

def set_track_area(ixx,  iyy, fxx,  fyy):
    global ix,  iy,  fx, fy, upper_left_x,  lower_right_x,  upper_left_y,  lower_right_y, monitored_width,  monitored_height
    ix=ixx
    iy=iyy
    fx=fxx
    fy=fyy
    if fx > ix:
        upper_left_x = ix
        lower_right_x = fx
    else:
        upper_left_x = fx
        lower_right_x = ix

    if fy > iy:
        upper_left_y = iy
        lower_right_y = fy
    else:
        upper_left_y = fy
        lower_right_y = iy
    monitored_width = lower_right_x - upper_left_x
    monitored_height = lower_right_y - upper_left_y

image_w=640
image_h=480
image_d=3
framerate = 20
iCam = 0
cam = 0

#import pygame
#import pygame.camera
#rPi 
from picamera.array import PiRGBArray
from picamera import PiCamera

# initialize the camera. Adjust vflip and hflip to reflect your camera's orientation
def setup_camera():
        global use_webcam,  cam,  rawCapture,  RESOLUTION,  FPS,  snapshot,  snapshot1
        if use_webcam:
            #global pygame
            cam = 0
            pygame.init()
            pygame.camera.init()
            camList = pygame.camera.list_cameras()
            print('cameras', camList)
            cam = pygame.camera.Camera(camList[cam], RESOLUTION, "RGB")
            cam.start()
            #framerate = framerate
            print('WebcamVideoStream loaded.. .warming camera')
            time.sleep(2)
        else:
            global picamera
            # initialize the camera. Adjust vflip and hflip to reflect your camera's orientation
            cam = PiCamera()
            cam.resolution = RESOLUTION
            cam.framerate = FPS
            cam.vflip = False
            cam.hflip = False
            rawCapture = PiRGBArray(cam, size=cam.resolution)
            # allow the camera to warm up
            time.sleep(0.9)

# setup headless, config etc on startup
initialize()
setup_camera()

#import pygame.image
#djl added if web-cam
if use_webcam:
  if cam.query_image():
  #    global snapshot
  #    global snapshot1
  #    global frame
    # snapshot = self.cam.get_image()
    # self.frame = list(pygame.image.tostring(snapshot, "RGB", False))
    snapshot = cam.get_image()
    snapshot1 = pygame.transform.scale(snapshot, RESOLUTION)
    frame = pygame.surfarray.pixels3d(pygame.transform.rotate(pygame.transform.flip(snapshot1, False, False), 90))

#
# HERE IS SOME GUI STUFF
#
# create an image window and place it in the upper left corner of the screen
#if SHOW_IMAGE == True:
if headless == False:
    cv2.namedWindow("Speed Camera")
    cv2.moveWindow("Speed Camera", 10, 40)

    # call the draw_rectangle routines when the mouse is used
    cv2.setMouseCallback('Speed Camera',draw_rectangle)
 
camera = cam  #djl TODO: fix and go back to using "camera"
image = None #djl 
org_image = None #djl 
#djl commented out rawCapture = snapshot
#rawCapture = snapshot
if use_webcam:
    # grab a reference image to use for drawing the monitored area's boundry
    img = cam.get_image()
    img1 = pygame.transform.scale(img, RESOLUTION)
    image = pygame.surfarray.pixels3d(pygame.transform.rotate(pygame.transform.flip(img1, False, False), 90))
    # for pygame, the copy function makes the image numpy compatible
    image = image.copy()
    org_image = image.copy()
else:
    cam.capture(rawCapture, format="bgr", use_video_port=True)
    image = rawCapture.array
    rawCapture.truncate(0)
    org_image = image.copy()



# added section for saving of the monitoriing area
if SAVE_CSV:
    csvfileout = "carspeed_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    record_speed('Date,Day,Time,Speed,Image')
else:
    csvfileout = ''

# preload config_file monitored area data if config_file defined, should be loaded by now
prompt = "Define the monitored area - press 'c' to continue" 
if config_file != None :
    image = org_image.copy()
    if headless == False:
        prompt_on_image(prompt)
        cv2.rectangle(image,(ix,iy),(fx,fy),(0,255,0),2)
    setup_complete=True
else:
    if headless == False:
        prompt_on_image(prompt)    
 
# wait while the user draws the monitored area's boundry
while not setup_complete:
    cv2.imshow("Speed Camera",image)
 
    #wait for for c to be pressed  
    key = cv2.waitKey(1) & 0xFF
  
    # if the `c` key is pressed, break from the loop
    if key == ord("c"):
        break

# the monitored area is defined, time to move on
prompt = "Press 'q' to quit" 
 
# since the monitored area's bounding box could be drawn starting 
# from any corner, normalize the coordinates
 
if fx > ix:
    upper_left_x = ix
    lower_right_x = fx
else:
    upper_left_x = fx
    lower_right_x = ix
 
if fy > iy:
    upper_left_y = iy
    lower_right_y = fy
else:
    upper_left_y = fy
    lower_right_y = iy
     
monitored_width = lower_right_x - upper_left_x
monitored_height = lower_right_y - upper_left_y

if SAVE_CSV and config_file == None:
    csvmonfileout = "monitoringarea_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    record_mon_area('UpperLeftX,UpperLeftY,LowerRightX,LowerRightY,MonitoredWidth,MonitoredHeight')
    record_mon_area(("%d" % upper_left_x)+','+("%d" % upper_left_y)+','+("%d" % lower_right_x)+','+("%d" % lower_right_y)+','+\
        ("%d" % monitored_width)+','+("%d" % monitored_height))
else:
    csvmonfileout = '' 

# capture frames from the camera (using capture_continuous.
#   This keeps the picamera in capture mode - it doesn't need
#   to prep for each frame's capture.
#
#rPi
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#while True: #for web_cam
    if use_webcam:
        img = cam.get_image()
        img1 = pygame.transform.scale(img, RESOLUTION)

    #initialize the timestamp
    timestamp = datetime.datetime.now()
 
    # grab the raw NumPy array representing the image 
    if use_webcam:
        image = pygame.surfarray.pixels3d(pygame.transform.rotate(pygame.transform.flip(img1, False, False), 90))
        image = image.copy()
    else:
        image = frame.array
 
    # crop area defined by [y1:y2,x1:x2]
    gray = image[upper_left_y:lower_right_y,upper_left_x:lower_right_x]
 
    # convert the fram to grayscale, and blur it
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLURSIZE, 0)
 
    # if the base image has not been defined, initialize it
    if base_image is None:
        base_image = gray.copy().astype("float")
        lastTime = timestamp
        if not use_webcam:
            rawCapture.truncate(0)
        if headless == False:
            cv2.imshow("Speed Camera", image)
  
    # compute the absolute difference between the current image and
    # base image and then turn eveything lighter gray than THRESHOLD into
    # white
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(base_image))
    thresh = cv2.threshold(frameDelta, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    
    # dilate the thresholded image to fill in any holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # look for motion 
    motion_found = False
    biggest_area = 0
 
    # examine the contours, looking for the largest one
    for c in cnts:
        (x1, y1, w1, h1) = cv2.boundingRect(c)
        # get an approximate area of the contour
        found_area = w1*h1 
        # find the largest bounding rectangle
        if (found_area > MIN_AREA) and (found_area > biggest_area):  
            biggest_area = found_area
            motion_found = True
            x = x1
            y = y1
            h = h1
            w = w1

    if motion_found:
        if state == WAITING:
            # intialize tracking
            state = TRACKING
            initial_x = x
            last_x = x
            initial_time = timestamp
            last_mph = 0
            text_on_image = 'Tracking'
            print(text_on_image)
            print("x-chg    Secs      MPH  x-pos width")
        else:
            # compute the lapsed time
            secs = secs_diff(timestamp,initial_time)

            if secs >= 15:
                state = WAITING
                direction = UNKNOWN
                text_on_image = 'No Car Detected'
                motion_found = False
                biggest_area = 0
                #djl
                if not use_webcam:
                    rawCapture.truncate(0)
                base_image = None
                print('Resetting')
                continue             

            if state == TRACKING:       
                if x >= last_x:
                    direction = LEFT_TO_RIGHT
                    abs_chg = x + w - initial_x
                else:
                    direction = RIGHT_TO_LEFT
                    abs_chg = initial_x - x
                mph = get_speed(abs_chg,ftperpixel,secs)
                print("{0:4d}  {1:7.2f}  {2:7.0f}   {3:4d}  {4:4d}".format(abs_chg,secs,mph,x,w))
                real_y = upper_left_y + y
                real_x = upper_left_x + x
                # is front of object outside the monitired boundary? Then write date, time and speed on image
                # and save it 
                if ((x <= 2) and (direction == RIGHT_TO_LEFT)) \
                        or ((x+w >= monitored_width - 2) \
                        and (direction == LEFT_TO_RIGHT)):
                    if (last_mph > MIN_SPEED):    # save the image
                        # timestamp the image
                        cv2.putText(image, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                            (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
                        # write the speed: first get the size of the text
                        size, base = cv2.getTextSize( "%.0f mph" % last_mph, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                        # then center it horizontally on the image
                        cntr_x = int((IMAGEWIDTH - size[0]) / 2) 
                        cv2.putText(image, "%.0f mph" % last_mph,
                            (cntr_x , int(IMAGEHEIGHT * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 2.00, (0, 255, 0), 3)
                        # and save the image to disk
                        imageFilename = "car_at_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                        # use the following image file name if you want to be able to sort the images by speed
                        #imageFilename = "car_at_%02.0f" % last_mph + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                        
                        cv2.imwrite(imageFilename,image)
                        if SAVE_CSV:
                            cap_time = datetime.datetime.now()
                            record_speed(cap_time.strftime("%Y.%m.%d")+','+cap_time.strftime('%A')+','+\
                               cap_time.strftime('%H%M')+','+("%.0f" % last_mph) + ','+imageFilename)
                    state = SAVING
                # if the object hasn't reached the end of the monitored area, just remember the speed 
                # and its last position
                last_mph = mph
                last_x = x
    else:
        if state != WAITING:
            state = WAITING
            direction = UNKNOWN
            text_on_image = 'No Car Detected'
            print(text_on_image)
            
    # only update image and wait for a keypress when waiting for a car
    # This is required since waitkey slows processing.
    if (state == WAITING):    
 
        # draw the text and timestamp on the frame
        cv2.putText(image, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        cv2.putText(image, "Road Status: {}".format(text_on_image), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,0.35, (0, 0, 255), 1)
     
        if SHOW_BOUNDS:
            #define the monitored area right and left boundary
            cv2.line(image,(upper_left_x,upper_left_y),(upper_left_x,lower_right_y),(0, 255, 0))
            cv2.line(image,(lower_right_x,upper_left_y),(lower_right_x,lower_right_y),(0, 255, 0))
       
        # show the frame and check for a keypress
        if SHOW_IMAGE:
            prompt_on_image(prompt)
            cv2.imshow("Speed Camera", image)
            
        # Adjust the base_image as lighting changes through the day
        if state == WAITING:
            last_x = 0
            cv2.accumulateWeighted(gray, base_image, 0.25)
 
        state=WAITING;
        key = cv2.waitKey(1) & 0xFF
      
        # if the `q` key is pressed, break from the loop and terminate processing
        if key == ord("q"):
            break
         
    # clear the stream in preparation for the next frame
    if not use_webcam:
        rawCapture.truncate(0)
  
# cleanup the camera and close any open windows
cv2.destroyAllWindows()

