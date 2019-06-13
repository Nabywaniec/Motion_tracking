import time
from detector import Detector
from comparator import Comparator
import logging

DELAY = 0.
previous_objects = None

appLogger = logging.getLogger(__name__)


def configure_logger():
    appLogger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [YoloDetector] %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    appLogger.addHandler(ch)


def detect_objects():
    global previous_objects, appLogger
    detector = Detector()
    comp = Comparator(detector)
    while True:
        detected_objects = detector.process_next_frame()
        print("[Detection Stage] Done")
        if not previous_objects:
            previous_objects = detected_objects
        else:
            print("[Looking for changes - Tracking Stage]")
            print(comp.compare_objects_lists(previous_objects, detected_objects))
            previous_objects = detected_objects
        detector.show_frame()
        time.sleep(DELAY)


if __name__ == '__main__':
    configure_logger()
    detect_objects()
