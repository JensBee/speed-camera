"""
Return the estimated distance based on a real event and the actual kph

Usage:
    calibrate.py <eventfile> --kph=<kph>

Options:
    -h --help     Show this screen.
"""

import logging
import json
import docopt
import math
import pathlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)
# Load arguments
args = docopt.docopt(__doc__)

# Get the actual speed
speed = int(args['--speed'])

# Load the event data
event_data = None
with open(pathlib.Path(args['<eventfile>']), 'r') as json_file:
  event_data = json.load(json_file)

# Figure out the speed
total_distance = 0
direction = None
for frame, event in enumerate(event_data):
    direction = event['dir']
    # MATH!
    distance = ((((speed / 3.6) * event['secs']) / event['delta']) * float(event['image_width'])) / (2 * (math.tan(math.radians(event['fov'] * 0.5))))
    total_distance += distance
    logging.info('Frame {:2.0f}: {:1.2f}sec {:4.2f}px {:2.0f}kph == {:4.2f} distance'.format(frame, event['secs'], event['delta'], event['speed'], distance))

logging.info('Updated distance for {}: {:4.2f}'.format(direction, total_distance / len(event_data)))
