# Car Speed Detection - speed-camera.py

## Description

![example-result](https://user-images.githubusercontent.com/622065/88995677-207ec280-d2a0-11ea-9cc0-4f8271f7de09.gif)

This program, designed for the Raspberry Pi, records the speed of cars driving perpendicular to a PiCamera.  For vehicles traveling faster than a specified speed, it will send the video feed via Telegram.  It determines speed based on the field-of-view of the camera and distance from the cars.

All of the data is output in a CSV format that works well in software like Splunk.

## Requirements

* Raspberry Pi 2 Model B (or 3)
* Picamera
* Opencv
* Imagemagik
* Python 3

## Configuration

There are a number of different sections in the `config.yaml`

### Monitoring Location

| *field* | *default* | *description* |
| ------- | --------- | ------------- |
| `upper_left_x` | `0` | top left X-value of the image monitoring area |
| `upper_left_y` | `0` | top left Y-value of the image monitoring area |
| `lower_right_x` | `1024` | bottom right X-value of the image monitoring area |
| `lower_right_y` | `576` | bottom right Y-value of the image monitoring area |
| `l2r_distance` | `65` | distance (in ft) from camera to car traveling left to right |
| `r2l_distance` | `80` | distance (in ft) from camera to car traveling right to left |

### Camera Settings

For this section, I recommend reviewing reviewing the [Hardware - Sensor Modes](https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes) section of the PiCamera docs.  The default specified here works well with a PiCamera v2.

| *field* | *default* | *description* |
| ------- | --------- | ------------- |
| `fov` | `62.2` | field of view for this camera model |
| `fps` | `30` | desired FPS (not actually) |
| `image_width` | `1024` | width of image |
| `image_height` | `576` | height of image |
| `image_min_area` | `500` | minimum determined area that _could_ be a car |

### Thresholds

For this section, speed is the mean MPH, area is pixels, and confidence is how close the standard deviation of recorded MPH is from the mean.

| *field* | *default* | *description* |
| ------- | --------- | ------------- |
| `min_speed` | `10` | minimum speed to record to the logs |
| `min_speed_alert` | `30` | minimum speed to send an alert |
| `min_area` | `2000` | minimum area to record to the logs |
| `min_confidence` | `70` | minimum confidence to record to the logs |
| `min_confidence_alert` | `90` | minimum confidence to send an alert |
| `min_distance` | `0.4` | minimum seconds between events |

### Communication

| *field* | *default* | *description* |
| ------- | --------- | ------------- |
| `telegram_token` | `None` | bot token to authenticate with Telegram |
| `telegram_chat_id` | `None` | person/group `chat_id` to send the alert to |
| `telegram_frequency` | `6` | hours between periodic text updates |

## Installation

1. Copy all files to the Pi under `/home/pi/speed-camera`
2. Install dependencies `$ sudo make install`
3. Create a config file at `config.yaml` (see *Configuration*)
4. Start the service `$ sudo make restart`
5. Tail the logs `$ make tail`

## Usage

### Preview

Preview mode writes a `preview.jpg` file out on disk as well as sending to Telegram.  This is useful for determining the `(X,Y)` coordinates you want to monitor.

![preview](https://user-images.githubusercontent.com/622065/88995809-781d2e00-d2a0-11ea-9096-7bf43b9f8120.png)

```
$ python3 speed-camera.py preview --config config.yaml
2020-07-30 09:47:40,960 Initializing
2020-07-30 09:47:40,990 Booting up camera
2020-07-30 09:47:43,087 Monitoring: (90,292) to (1000,496) = 910x204 space
2020-07-30 09:47:43,090 L2R: 45ft from camera == 0.05 per pixel
2020-07-30 09:47:43,091 R2L: 45ft from camera == 0.05 per pixel
```

### Normal

During normal operation, the service will write all logs to `logs/service.log`, metrics to `logs/recorded_speed.csv`, and any "alerts" to `logs/YYYY-MM-DD_HH:MM:SS.SSSSSS-SPEEDmph-CONFIDENCE.json` and `logs/YYYY-MM-DD_HH:MM:SS.SSSSSS-SPEEDmph-CONFIDENCE.gif`.

```
$ python3 speed-camera.py --config config.yaml
2020-07-30 09:47:40,960 Initializing
2020-07-30 09:47:40,990 Booting up camera
2020-07-30 09:47:43,087 Monitoring: (90,292) to (1000,496) = 910x204 space
2020-07-30 09:47:43,090 L2R: 45ft from camera == 0.05 per pixel
2020-07-30 09:47:43,091 R2L: 45ft from camera == 0.05 per pixel
2020-07-30 09:49:34,731 Tracking
2020-07-30 09:49:34,732 Initial Data: x=871 w=39 area=2730 gap=111.603324
2020-07-30 09:49:34,732  x-Δ     Secs      MPH  x-pos width area dir
2020-07-30 09:49:34,834   55     0.10       20    816    94 9306 RTL
2020-07-30 09:49:34,968  128     0.23       20    743   167 19706 RTL
2020-07-30 09:49:35,146  203     0.38       20    668   242 30492 RTL
2020-07-30 09:49:35,268  296     0.53       20    575   335 44220 RTL
2020-07-30 09:49:35,401  363     0.67       20    508   388 52380 RTL
2020-07-30 09:49:35,534  436     0.80       20    435   379 50407 RTL
2020-07-30 09:49:35,668  508     0.93       20    363   369 47970 RTL
2020-07-30 09:49:35,801  583     1.07       20    288   362 45612 RTL
2020-07-30 09:49:35,980  659     1.21       20    212   358 44392 RTL
2020-07-30 09:49:36,103  752     1.37       20    119   354 44250 RTL
2020-07-30 09:49:36,234  827     1.50       20     44   355 43665 RTL
2020-07-30 09:49:36,367  871     1.63       19      0   326 37816 RTL
2020-07-30 09:49:36,369 Determined area:   avg=42309 deviation=9349 frames=12
2020-07-30 09:49:36,370 Determined speed: mean=  20 deviation=   0 frames=12
2020-07-30 09:49:36,370 Overall Confidence Level 99%
2020-07-30 09:49:36,371 Event recorded
2020-07-30 09:49:36,999 Resetting
2020-07-30 09:49:43,894 Current FPS @ 8
...
```

### Calibration

Sometimes your distance isn't _just_ right.  So if you record an event and you know the actual speed, you can use the calibration tool to calculate the right distance.

```
$ python3 calibrate.py logs/2020-07-30_16:52:13.285767-21mph-99.json --mph=36
2020-07-30 19:07:33,141 Frame  0: 0.13sec 79.00px 21mph == 75.98 distance
2020-07-30 19:07:33,141 Frame  1: 0.27sec 156.00px 21mph == 76.64 distance
2020-07-30 19:07:33,141 Frame  2: 0.41sec 238.00px 21mph == 77.98 distance
2020-07-30 19:07:33,141 Frame  3: 0.57sec 338.00px 22mph == 75.23 distance
2020-07-30 19:07:33,141 Frame  4: 0.70sec 416.00px 21mph == 75.40 distance
2020-07-30 19:07:33,141 Frame  5: 0.83sec 494.00px 21mph == 75.59 distance
2020-07-30 19:07:33,141 Frame  6: 0.97sec 573.00px 21mph == 75.65 distance
2020-07-30 19:07:33,141 Frame  7: 1.10sec 654.00px 21mph == 75.36 distance
2020-07-30 19:07:33,141 Frame  8: 1.23sec 731.00px 21mph == 75.61 distance
2020-07-30 19:07:33,141 Frame  9: 1.37sec 813.00px 21mph == 75.64 distance
2020-07-30 19:07:33,141 Frame 10: 1.50sec 863.00px 21mph == 77.88 distance
2020-07-30 19:07:33,141 Updated distance for RTL: 76.09
```

In this example, the car was recorded at 21 MPH but it actually was going 36 MPH.  The output says to adjust the RTL distance to 76ft.

## References

- Original project: https://github.com/gregtinkers/carspeed.py
- Version 2 fork: https://github.com/dlarue/carspeed
- Version 3 fork: https://github.com/thesilentmiaow/carspeed.py

## License:

(The MIT License)

Copyright (c) 2016 Greg Barbu

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
