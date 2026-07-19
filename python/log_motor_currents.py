###############################################################################
# log_motor_currents.py
# Continuously logs all motor currents to a CSV file
#
# The CSV file will have the following columns:
#   timestamp (ms), module_id, motor_type, motor_id, current_ma
#
# Motor types: front_leg, back_leg, body
###############################################################################

import argparse
import csv
import os
import time
from datetime import datetime
from dataclasses import dataclass
from beco_v3 import Motor, Module, Robot

try:
    import cv2
except ImportError:
    cv2 = None

# --- LSS motor IDs ----------------------------------------------------------

LSS_ID_LEG_1  = 0
LSS_ID_LEG_2  = 2
LSS_ID_BODY_1 = 1

# --- Constants --------------------------------------------------------------
MOTOR_BREAKER_CURRENT = 2000  # mA
LOG_FOLDER = "logs"
VIDEO_FOLDER = "video_logs"
VIDEO_CAMERA_INDEX = 0
VIDEO_CAPTURE_BACKENDS = ['CAP_DSHOW', 'CAP_MSMF', 'CAP_ANY']
VIDEO_FOURCC = "mp4v"
VIDEO_FOURCC_FALLBACKS = ["avc1", "H264", "X264", "MJPG"]
DEFAULT_VIDEO_FPS = 30.0

# --- Trial Settings ----------------------------------------------------------------
SPRING_WIRE_DIAM = 1.0 # mm
SPACING = 20 # mm 

# --- ModuleConfig dataclass (replaces C++ struct) ---------------------------

@dataclass
class ModuleConfig:
    period:                   int
    body_amp:                 float
    phi_FB:                   float   # phase diff between front leg and body
    phi_FR:                   float   # phase diff between front and rear leg
    frontleg_standard_phase:  float
    servo_deg_scale:          float
    leg_amp:                  float
    angle_offset_leg:         float
    angle_offset_body:        float


def ensure_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def build_video_filename():
    return f"{SPRING_WIRE_DIAM}_{SPACING}_{module1.phi_FB}.mp4"


def initialize_video_capture(camera_index=VIDEO_CAMERA_INDEX):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video recording. Install opencv-python.")

    last_exception = None
    for backend_name in VIDEO_CAPTURE_BACKENDS:
        backend = getattr(cv2, backend_name, cv2.CAP_ANY)
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            print(f"Opened camera #{camera_index} using backend {backend_name}.")
            return cap
        cap.release()
        last_exception = backend_name

    raise RuntimeError(
        f"Unable to open camera #{camera_index}. Tried backends: {', '.join(VIDEO_CAPTURE_BACKENDS)}. "
        "Check that your built-in camera is available and not already in use."
    )


def initialize_video_writer(cap, video_path):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    if width <= 0 or height <= 0:
        width, height = 640, 480

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FPS
    if fps <= 0:
        fps = DEFAULT_VIDEO_FPS

    tried_codecs = []
    for codec in [VIDEO_FOURCC] + VIDEO_FOURCC_FALLBACKS:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        tried_codecs.append(codec)
        if writer.isOpened():
            print(f"Using video codec: {codec}")
            return writer, video_path
        writer.release()

    # Try an AVI fallback if MP4 codecs fail
    fallback_path = os.path.splitext(video_path)[0] + ".avi"
    for codec in ["XVID", "MJPG", "DIVX"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(fallback_path, fourcc, fps, (width, height))
        tried_codecs.append(codec)
        if writer.isOpened():
            print(f"Using fallback codec: {codec} with file: {fallback_path}")
            return writer, fallback_path
        writer.release()

    cap.release()
    raise RuntimeError(
        f"Unable to open any video writer for: {video_path}. Tried codecs: {', '.join(tried_codecs)}. "
        "Make sure your OpenCV build supports video encoding and that the output path is writable."
    )


def start_video_recording(video_path):
    ensure_folder(VIDEO_FOLDER)
    video_path = os.path.join(VIDEO_FOLDER, video_path)

    cap = initialize_video_capture()
    writer, actual_video_path = initialize_video_writer(cap, video_path)
    print(f"Recording video to: {actual_video_path}")

    return cap, writer, actual_video_path


def stop_video_recording(cap, writer):
    if writer is not None:
        writer.release()
    if cap is not None:
        cap.release()
    print("Video recording stopped.")


def record_frame(cap, writer):
    ret, frame = cap.read()
    if ret:
        writer.write(frame)
    else:
        print("Warning: failed to capture video frame.")

# --- Module 1 configuration -------------------------------------------------

module1 = ModuleConfig(
    period                  = 3000,
    body_amp                = 600,
    phi_FB                  = 0.34, # 0.11, 0.12, 0.44, 0.43/0.42
    phi_FR                  = 0.5,
    frontleg_standard_phase = 0,
    servo_deg_scale         = 10,
    leg_amp                 = 1100, # previously 1700
    angle_offset_leg        = 1200,
    angle_offset_body       = 0,
)

# --- Build motors -----------------------------------------------------------

front_leg = Motor(LSS_ID_LEG_1, module1.leg_amp, module1.period,
                  module1.angle_offset_leg, module1.frontleg_standard_phase)

back_leg  = Motor(LSS_ID_LEG_2, module1.leg_amp, module1.period,
                  module1.angle_offset_leg, module1.phi_FR)

body      = Motor(LSS_ID_BODY_1, module1.body_amp, module1.period,
                  module1.angle_offset_body, module1.phi_FB)

# --- Build module and robot -------------------------------------------------

module = Module(front_leg, back_leg, body,
                fb_offset=module1.phi_FB,
                fr_offset=module1.phi_FR,
                f_offset=0.0)

robot = Robot([module])

# --- CSV logging function ---------------------------------------------------

def log_motor_currents(robot, csv_filename, record=False, video_filename=None):
    """
    Continuously logs motor currents from the robot to a CSV file.
    Logs: timestamp, module_id, motor_type, motor_id, current_ma
    """
    ensure_folder(LOG_FOLDER)
    csv_path = os.path.join(LOG_FOLDER, csv_filename)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp_ms', 'module_id', 'motor_type', 'motor_id', 'current_ma']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        video_cap = None
        video_writer = None
        video_path = None
        if record and video_filename is not None:
            video_cap, video_writer, video_path = start_video_recording(video_filename)

        print(f"Logging motor currents to: {csv_path}")
        if record:
            print("Press Ctrl+C to stop logging and video recording.\n")
        else:
            print("Press Ctrl+C to stop logging.\n")

        try:
            while True:
                if not robot.loop(MOTOR_BREAKER_CURRENT):
                    break

                timestamp_ms = robot._millis() - robot.start_time

                for module_id, mod in enumerate(robot.modules):
                    motor_types = ['front_leg', 'back_leg', 'body']
                    motors = [mod.front_leg, mod.back_leg, mod.body]

                    for motor_type, motor in zip(motor_types, motors):
                        current_ma = motor.get_current()
                        writer.writerow({
                            'timestamp_ms': f"{timestamp_ms:.2f}",
                            'module_id': module_id,
                            'motor_type': motor_type,
                            'motor_id': motor.ID,
                            'current_ma': f"{current_ma:.2f}"
                        })

                if record:
                    record_frame(video_cap, video_writer)

                if int(timestamp_ms) % 5000 == 0:
                    print(f"Logging... {timestamp_ms/1000:.1f}s elapsed")

                time.sleep(0.02)

        except KeyboardInterrupt:
            pass
        finally:
            robot.go_limp()
            print("\n\nLogging stopped.")
            print(f"Data saved to: {csv_path}")
            if record and video_path is not None:
                stop_video_recording(video_cap, video_writer)
                print(f"Video saved to: {video_path}")


# --- Entry point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log motor currents with optional video recording.")
    parser.add_argument('-record', '--record', action='store_true', help='Enable video recording using the built-in camera')
    args = parser.parse_args()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"motor_currents_{timestamp_str}.csv"

    robot.setup()

    video_filename = build_video_filename() if args.record else None
    log_motor_currents(robot, csv_filename, record=args.record, video_filename=video_filename)
