# CarSpeed Version 4.0
"""
Script to capture moving car speed

Usage:
    carspeed.py [preview] [--config=<file>]

Options:
    -h --help     Show this screen.
"""

# import the necessary packages
from docopt import docopt
from picamera import PiCamera
from picamera.array import PiRGBArray
from pathlib import Path
import cv2
import datetime
import numpy as np
import logging
import time
import math
import json
import yaml
import telegram
import subprocess

# Location for files/logs
FILENAME_SERVICE = "logs/service.log"
FILENAME_RECORD = "logs/recorded_speed.csv"

# Important constants
MIN_SAVE_BUFFER = 2
THRESHOLD = 25
BLURSIZE = (15,15)
RECORD_HEADERS = 'Timestamp,Speed,SpeedDeviation,Area,AreaDeviation,Frames,Seconds,Direction'

# the following enumerated values are used to make the program more readable
WAITING = 0
TRACKING = 1
SAVING = 2
UNKNOWN = 0
LEFT_TO_RIGHT = 1
RIGHT_TO_LEFT = 2

class Config:
    # monitoring area
    upper_left_x = 0
    upper_left_y = 0
    lower_right_x = 1024
    lower_right_y = 576
    # range
    l2r_distance = 65     # <---- distance-to-road in feet (left-to-right side)
    r2l_distance = 80     # <---- distance-to-road in feet (right-to-left side)
    # camera settings
    fov = 62.2            # <---- field of view
    fps = 30              # <---- frames per second
    image_width = 1024    # <---- resolution width
    image_height = 576    # <---- resolution height
    # thresholds for recording
    too_close = 0.4       # <----
    min_speed_save = 10   # <---- minimum speed for saving records
    min_speed_image  = 30 # <---- minimum speed for saving images
    min_area_detect = 500 # <---- minimum area for detecting motion
    min_area_save = 2000  # <---- minimum area for saving records
    # communication
    telegram_token = ""   # <----
    telegram_chat_id = "" # <----
    # debug
    debug_enabled = False  # <----

    @staticmethod
    def load(config_file):
        cfg = Config()
        with open(config_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)

                for key, value in data.items():
                    if hasattr(cfg, key):
                        setattr(cfg, key, value)

            except yaml.YAMLError as exc:
                logging.error("Failed to load config: {}".format(exc))
                exit(1)

        # Swap positions
        if cfg.upper_left_x > cfg.lower_right_x:
            cfg.upper_left_x = cfg.lower_right_x
            cfg.lower_right_x = cfg.upper_left_x

        if cfg.upper_left_y > cfg.lower_right_y:
            cfg.upper_left_y = cfg.lower_right_y
            cfg.lower_right_y = cfg.upper_left_y

        cfg.upper_left = (cfg.upper_left_x, cfg.upper_left_y)
        cfg.lower_right = (cfg.lower_right_x, cfg.lower_right_y)

        cfg.monitored_width = cfg.lower_right_x - cfg.upper_left_x
        cfg.monitored_height = cfg.lower_right_y - cfg.upper_left_y

        cfg.resolution = [cfg.image_width, cfg.image_height]

        return cfg

# calculate speed from pixels and time
def get_speed(pixels, ftperpixel, secs):
    if secs > 0.0:
        return ((pixels * ftperpixel)/ secs) * 0.681818
    else:
        return 0.0

# calculate pixel width
def get_pixel_width(fov, distance, image_width):
    frame_width_ft = 2 * (math.tan(math.radians(fov * 0.5)) * distance)
    ft_per_pixel = frame_width_ft / float(image_width)

    return ft_per_pixel

def str_direction(direction):
    if direction == LEFT_TO_RIGHT:
        return "LTR"
    elif direction == RIGHT_TO_LEFT:
        return "RTL"
    else:
        return "???"

# calculate elapsed seconds
def secs_diff(endTime, begTime):
    diff = (endTime - begTime).total_seconds()
    return diff

# record speed in .csv format
def save_record(res):
    global FILENAME_RECORD
    f = open(FILENAME_RECORD, 'a')
    f.write(res+"\n")
    f.close

def parse_command_line():
    preview = False
    config_file = None

    logging.info("Initializing")
    args = docopt(__doc__)

    if args['preview']:
        preview=True

    if args['--config']:
        config_file = Path(args['--config'])
        if not config_file.is_file():
            logging.error("config file does NOT exist")
            exit(1)

    return (preview, config_file)

def detect_motion(image, min_area):
    # dilate the thresholded image to fill in any holes, then find contours
    # on thresholded image
    image = cv2.dilate(image, None, iterations=2)
    (_, cnts, _) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # look for motion
    motion_found = False
    biggest_area = 0
    x = 0
    y = 0
    w = 0
    h = 0

    # examine the contours, looking for the largest one
    for c in cnts:
        (x1, y1, w1, h1) = cv2.boundingRect(c)
        # get an approximate area of the contour
        found_area = w1 * h1
        # find the largest bounding rectangle
        if (found_area > min_area) and (found_area > biggest_area):
            biggest_area = found_area
            motion_found = True
            x = x1
            y = y1
            w = w1
            h = h1

    return (motion_found, x, y, w, h, biggest_area)

def annotate_image(image, timestamp, mph=0, confidence=0):
    global cfg

    # make it gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # timestamp the image
    cv2.putText(image, timestamp.strftime("%d %B %Y %H:%M:%S.%f"),
                (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # write the speed
    if mph > 0:
        msg = "{:.0f} mph".format(mph)
        (size, _) = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)

        # then center it horizontally on the image
        cntr_x = int((cfg.image_width - size[0]) / 2)
        cv2.putText(image, msg, (cntr_x, int(cfg.image_height * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 2.00, (0, 0, 255), 3)

    # write the confidence
    if confidence > 0:
        msg = "{:.0f}%".format(confidence)
        (size, _) = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)

        # then right align it horizontally on the image
        cntr_x = int((cfg.image_width - size[0]) / 4) * 3
        cv2.putText(image, msg, (cntr_x, int(cfg.image_height * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 1.00, (0, 0, 255), 3)

    # define the monitored area right and left boundary
    cv2.line(image, (cfg.upper_left_x, cfg.upper_left_y),
                (cfg.upper_left_x, cfg.lower_right_y), (0, 255, 0), 4)
    cv2.line(image, (cfg.lower_right_x, cfg.upper_left_y),
                (cfg.lower_right_x, cfg.lower_right_y), (0, 255, 0), 4)

    return image


def save_debug(timestamp, events):
    global cfg

    folder = "logs/debug/{}".format(timestamp)
    Path(folder).mkdir(parents=True, exist_ok=True)

    data = []
    for e in events:
        # annotate it
        image = annotate_image(e['image'], e['ts'], mph=e['mph'])

        # Add the boundary
        cv2.rectangle(image,
            (cfg.upper_left_x + e['x'], cfg.upper_left_y + e['y']),
            (cfg.upper_left_x + e['x'] + e['w'], cfg.upper_left_y + e['y'] + e['h']), (0, 255, 0), 2)

        # and save the image to disk
        imageFilename = "{}/{}.jpg".format(folder, e['ts'])
        cv2.imwrite(imageFilename, image)

        del(e['image'])
        e['ts'] = e['ts'].timestamp()
        data.append(e)

    with open("{}/data.json".format(folder), 'w') as outfile:
        json.dump(data, outfile)

    # Create a gif
    subprocess.Popen(["/usr/bin/convert", "-delay", "10", "*.jpg", "animation.gif"], cwd=folder)

def save_image(image, timestamp, mph=0, confidence=0):
    global cfg, bot
    image = annotate_image(image, timestamp, mph=mph, confidence=confidence)

    # and save the image to disk
    imageFilename = "logs/car_at_{:02.0f}_{}.jpg".format(
        mph, timestamp.timestamp())
    cv2.imwrite(imageFilename, image)

    # send message to telegram
    if bot:
        bot.send_photo(
            chat_id=cfg.telegram_chat_id,
            photo=open(imageFilename, 'rb'),
            caption='{:.0f}mph @ {:.0f}%'.format(mph, confidence)
        )

# initialize the camera. Adjust vflip and hflip to reflect your camera's orientation
def setup_camera(cfg):
    logging.info("Booting up camera")

    # initialize the camera. Adjust vflip and hflip to reflect your camera's orientation
    camera = PiCamera(resolution=cfg.resolution, framerate=cfg.fps, sensor_mode=5)
    camera.vflip = False
    camera.hflip = False

    # start capturing
    capture = PiRGBArray(camera, size=camera.resolution)

    # allow the camera to warm up
    time.sleep(2)

    return (camera, capture)

# Setup logging
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(FILENAME_SERVICE),
        logging.StreamHandler()
    ]
)

# parse command-line
(PREVIEW, config_file) = parse_command_line()

# load config
cfg = Config.load(config_file)

# initialize messaging
bot = None
if cfg.telegram_token and cfg.telegram_chat_id:
    bot = telegram.Bot(cfg.telegram_token)

# setup camera
(camera, capture) = setup_camera(cfg)

# determine the boundary
logging.info("Monitoring: ({},{}) to ({},{}) = {}x{} space".format(
    cfg.upper_left_x, cfg.upper_left_y, cfg.lower_right_x, cfg.lower_right_y, cfg.monitored_width, cfg.monitored_height))

# calculate the the width of the image at the distance specified
l2r_ft_per_pixel = get_pixel_width(cfg.fov, cfg.l2r_distance, cfg.image_width)
r2l_ft_per_pixel = get_pixel_width(cfg.fov, cfg.r2l_distance, cfg.image_width)
logging.info("L2R: {:.0f}ft from camera == {:.2f} per pixel".format(cfg.l2r_distance, l2r_ft_per_pixel))
logging.info("R2L: {:.0f}ft from camera == {:.2f} per pixel".format(cfg.r2l_distance, r2l_ft_per_pixel))

# write headers to the output log
csv_file=Path(FILENAME_RECORD)
if not csv_file.is_file():
    save_record(RECORD_HEADERS)

state = WAITING
direction = UNKNOWN
# location
initial_x = 0
initial_w = 0
last_x = 0
last_w = 0
biggest_area = 0
areas = np.array([])
# timing
initial_time = datetime.datetime.now()
cap_time = datetime.datetime.now()
timestamp = datetime.datetime.now()
# speeds
sd = 0
speeds = np.array([])
counter = 0
# debug captures
debugs = []
# fps
fps_time = datetime.datetime.now()
fps_frames = 0
# capture
base_image = None
# stats
stats_l2r = np.array([])
stats_r2l = np.array([])
stats_time = datetime.datetime.now()

# capture frames from the camera (using capture_continuous.
#   This keeps the picamera in capture mode - it doesn't need
#   to prep for each frame's capture.
#
for frame in camera.capture_continuous(capture, format="bgr", use_video_port=True):
    # initialize the timestamp
    timestamp = datetime.datetime.now()

    # Save a preview of the image
    if PREVIEW:
        image = annotate_image(frame.array, timestamp)
        cv2.imwrite("preview.jpg", image)
        if bot:
            bot.send_photo(
                chat_id=cfg.telegram_chat_id,
                photo=open('preview.jpg', 'rb'),
                caption='Preview'
            )
        logging.info("Wrote preview.jpg")
        exit(0)

    # Log the current FPS
    fps_frames += 1
    if fps_frames > 1000:
        elapsed = secs_diff(timestamp, fps_time)
        logging.info("Current FPS @ {:.0f}".format(fps_frames/elapsed))
        fps_time = timestamp
        fps_frames = 0

    # Share stats every hour
    if secs_diff(timestamp, stats_time) > 60 * 60:
        stats_time = timestamp
        total = len(stats_l2r) + len(stats_r2l)
        if total > 0:
            l2r_perc = len(stats_l2r) / total * 100
            r2l_perc = len(stats_r2l) / total * 100

            l2r_mean = 0
            r2l_mean = 0
            if len(stats_l2r) > 0:
                l2r_mean = np.mean(stats_l2r)
            if len(stats_r2l) > 0:
                r2l_mean = np.mean(stats_r2l)

            bot.send_message(
                chat_id=cfg.telegram_chat_id,
                text="{:.0f} cars in the past hour\nL2R {:.0f}% at {:.0f} speed\nR2L {:.0f}% at {:.0f} speed".format(
                    total, l2r_perc, l2r_mean, r2l_perc, r2l_mean
                )
            )

        # clear stats
        stats_l2r = np.array([])
        stats_r2l = np.array([])
        stats_time = timestamp

    # grab the raw NumPy array representing the image
    image = frame.array

    # crop area defined by [y1:y2,x1:x2]
    gray = image[
        cfg.upper_left_y:cfg.lower_right_y,
        cfg.upper_left_x:cfg.lower_right_x
    ]

    # convert the fram to grayscale, and blur it
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLURSIZE, 0)

    # if the base image has not been defined, initialize it
    if base_image is None:
        base_image = gray.copy().astype("float")
        lastTime = timestamp
        capture.truncate(0)
        continue

    # compute the absolute difference between the current image and
    # base image and then turn eveything lighter gray than THRESHOLD into
    # white
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(base_image))
    thresh = cv2.threshold(frameDelta, THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # look for motion in the image
    (motion_found, x, y, w, h, biggest_area) = detect_motion(thresh, cfg.min_area_detect)

    if motion_found:
        if state == WAITING:
            # intialize tracking
            state = TRACKING
            initial_x = x
            initial_w = w
            last_x = x
            last_w = w
            initial_time = timestamp

            last_mph = 0

            # initialise array for storing speeds & standard deviation
            areas = np.array([])
            speeds = np.array([])

            # debug capturing
            debugs = []

            # detect gap and data points
            car_gap = secs_diff(initial_time, cap_time)

            logging.info('Tracking')
            logging.info("Initial Data: x={:.0f} w={:.0f} area={:.0f} gap={}".format(initial_x, initial_w, biggest_area, car_gap))
            logging.info(" x-Δ     Secs      MPH  x-pos width area dir")

            # if gap between cars too low then probably seeing tail lights of current car
            # but I might need to tweek this if find I'm not catching fast cars
            if (car_gap < cfg.too_close):
                state = WAITING
                direction = UNKNOWN
                motion_found = False
                biggest_area = 0
                capture.truncate(0)
                base_image = None
                logging.info("Car too close, skipping")
                continue
        else:
            # compute the lapsed time
            secs = secs_diff(timestamp, initial_time)

            # timeout after 5 seconds of inactivity
            if secs >= 5:
                state = WAITING
                direction = UNKNOWN
                motion_found = False
                biggest_area = 0
                capture.truncate(0)
                base_image = None
                logging.info('Resetting')
                continue

            if state == TRACKING:
                abs_chg = 0
                mph = 0
                if x >= last_x:
                    direction = LEFT_TO_RIGHT
                    abs_chg = (x + w) - (initial_x + initial_w)
                    mph = get_speed(abs_chg, l2r_ft_per_pixel, secs)
                else:
                    direction = RIGHT_TO_LEFT
                    abs_chg = initial_x - x
                    mph = get_speed(abs_chg, r2l_ft_per_pixel, secs)

                speeds = np.append(speeds, mph)
                areas = np.append(areas, biggest_area)

                # Store data
                if cfg.debug_enabled:
                    debugs.append({
                        'image': image.copy(),
                        'ts': timestamp,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'delta': abs_chg,
                        'area': biggest_area,
                        'mph': mph,
                        'dir': str_direction(direction),
                    })

                # If we've stopped or are going backward, reset.
                if mph <= 0:
                    logging.info("negative speed - stopping tracking")
                    if direction == LEFT_TO_RIGHT:
                        direction = RIGHT_TO_LEFT  # Reset correct direction
                        x = 1 # Force save
                    else:
                        direction = LEFT_TO_RIGHT  # Reset correct direction
                        x = cfg.monitored_width + MIN_SAVE_BUFFER  # Force save

                logging.info("{0:4d}  {1:7.2f}  {2:7.0f}   {3:4d}  {4:4d} {5:4d} {6:s}".format(
                    abs_chg, secs, mph, x, w, biggest_area, str_direction(direction)))

                # is front of object outside the monitired boundary? Then write date, time and speed on image
                # and save it
                if ((x <= MIN_SAVE_BUFFER) and (direction == RIGHT_TO_LEFT)) \
                        or ((x+w >= cfg.monitored_width - MIN_SAVE_BUFFER)
                        and (direction == LEFT_TO_RIGHT)):
                    sd_speed = 0
                    sd_area = 0
                    confidence = 0
                    #you need at least 3 data points to calculate a mean and we're deleting two
                    if (len(speeds) > 3):
                        # Mean of all items except the last one
                        mean_speed = np.mean(speeds[:-1])
                        # Mode of area (except the first and last)
                        avg_area = np.average(areas[1:-1])
                        # SD of all items except the last one
                        sd_speed = np.std(speeds[:-1])
                        sd_area = np.std(areas[1:-1])
                        confidence = ((mean_speed - sd_speed) / mean_speed) * 100
                    elif (len(speeds) > 1):
                        # use the last element in the array
                        mean_speed = speeds[-1]
                        avg_area = areas[-1]
                        # Set it to a very high value to highlight it's not to be trusted.
                        sd_speed = 99
                        sd_area = 99999
                    else:
                        mean_speed = 0  # ignore it
                        avg_area = 0
                        sd_speed = 0
                        sd_area = 0

                    logging.info("Determined area:   avg={:4.0f} deviation={:4.0f} frames={:0d}".format(avg_area, sd_area, len(areas)))
                    logging.info("Determined speed: mean={:4.0f} deviation={:4.0f} frames={:0d}".format(mean_speed, sd_speed, len(speeds)))
                    logging.info("Overall Confidence Level {:.0f}%".format(confidence))

                    # If they are speeding, record the event and image
                    if (mean_speed >= cfg.min_speed_image and avg_area >= cfg.min_area_save):
                        save_image(image, timestamp, mph=mean_speed,
                                   confidence=confidence)
                    if (mean_speed >= cfg.min_speed_save and avg_area >= cfg.min_area_save):
                        save_record("{},{:.0f},{:.0f},{:.0f},{:.0f},{:d},{:.2f},{:s}".format(
                            timestamp.timestamp(), mean_speed, sd_speed, avg_area, sd_area, len(speeds), secs, str_direction(direction)))

                        if cfg.debug_enabled:
                            save_debug(timestamp, debugs)

                        if direction == LEFT_TO_RIGHT and confidence > 75:
                            stats_l2r = np.append(stats_l2r, mean_speed)
                        elif direction == RIGHT_TO_LEFT and confidence > 75:
                            stats_r2l = np.append(stats_r2l, mean_speed)
                    else:
                        logging.info("Event not recorded: Speed or Area too low")

                    state = SAVING
                    cap_time = timestamp
                # if the object hasn't reached the end of the monitored area, just remember the speed
                # and its last position
                last_mph = mph
                last_x = x
    else:
        if state != WAITING:
            state = WAITING
            direction = UNKNOWN
            logging.info('Resetting')

    # Adjust the base_image as lighting changes through the day
    if state == WAITING:
        last_x = 0
        cv2.accumulateWeighted(gray, base_image, 0.25)

    # clear the stream in preparation for the next frame
    capture.truncate(0)

# cleanup the camera and close any open windows
cv2.destroyAllWindows()
