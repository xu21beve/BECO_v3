###############################################################################
# log_motor_currents.py
# Continuously logs all motor currents to a CSV file
#
# The CSV file will have the following columns:
#   timestamp (ms), module_id, motor_type, motor_id, current_ma
#
# Motor types: front_leg, back_leg, body
###############################################################################

import csv
import time
import os
from datetime import datetime
from beco_v3 import Motor, Module, Robot
from dataclasses import dataclass

# --- LSS motor IDs ----------------------------------------------------------

LSS_ID_LEG_1  = 2
LSS_ID_LEG_2  = 0
LSS_ID_BODY_1 = 1

# --- Constants --------------------------------------------------------------
MOTOR_BREAKER_CURRENT = 600  # mA

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

# --- Module 1 configuration -------------------------------------------------

module1 = ModuleConfig(
    period                  = 3000,
    body_amp                = 600,
    phi_FB                  = 0.7,
    phi_FR                  = 0.5,
    frontleg_standard_phase = 0,
    servo_deg_scale         = 10,
    leg_amp                 = 1300, # previously 1700
    angle_offset_leg        = 400,
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

def log_motor_currents(robot, csv_filename):
    """
    Continuously logs motor currents from the robot to a CSV file.
    Logs: timestamp, module_id, motor_type, motor_id, current_ma
    """
    csv_folder = "logs\\"

    # Create CSV file with header
    file_exists = os.path.isfile(csv_folder + csv_filename)
    
    with open(csv_folder + csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['timestamp_ms', 'module_id', 'motor_type', 'motor_id', 'current_ma']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
    
    print(f"Logging motor currents to: {csv_folder + csv_filename}")
    print("Press Ctrl+C to stop logging.\n")
    
    try:
        while True:
            # Get current time in milliseconds
            timestamp_ms = robot._millis() - robot.start_time
            
            # Log each module's motors
            for module_id, mod in enumerate(robot.modules):
                motor_types = ['front_leg', 'back_leg', 'body']
                motors = [mod.front_leg, mod.back_leg, mod.body]
                
                for motor_type, motor in zip(motor_types, motors):
                    current_ma = motor.get_current()
                    
                    with open(csv_folder + csv_filename, 'a', newline='') as csvfile:
                        fieldnames = ['timestamp_ms', 'module_id', 'motor_type', 'motor_id', 'current_ma']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        
                        writer.writerow({
                            'timestamp_ms': f"{timestamp_ms:.2f}",
                            'module_id': module_id,
                            'motor_type': motor_type,
                            'motor_id': motor.ID,
                            'current_ma': f"{current_ma:.2f}"
                        })
            
            # Small delay to avoid excessive logging (log every 20ms)
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        robot.go_limp()
        print("\n\nLogging stopped.")
        print(f"Data saved to: {csv_folder + csv_filename}")

# --- Entry point ---

if __name__ == "__main__":
    # Create CSV filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"logs\\motor_currents_{timestamp_str}.csv"
    
    # Setup robot and start logging
    robot.setup()
    
    try:
        while True:
            if not robot.loop(MOTOR_BREAKER_CURRENT):
                break
            
            # Log currents for all motors
            timestamp_ms = robot._millis() - robot.start_time
            
            for module_id, mod in enumerate(robot.modules):
                motor_types = ['front_leg', 'back_leg', 'body']
                motors = [mod.front_leg, mod.back_leg, mod.body]
                
                # with open(csv_filename, 'a', newline='') as csvfile:
                #     fieldnames = ['timestamp_ms', 'module_id', 'motor_type', 'motor_id', 'current_ma']
                #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                #     # Write header if file is new
                #     if os.path.getsize(csv_filename) == 0:
                #         writer.writeheader()
                    
                #     for motor_type, motor in zip(motor_types, motors):
                #         current_ma = motor.get_current()
                #         writer.writerow({
                #             'timestamp_ms': f"{timestamp_ms:.2f}",
                #             'module_id': module_id,
                #             'motor_type': motor_type,
                #             'motor_id': motor.ID,
                #             'current_ma': f"{current_ma:.2f}"
                #         })
            
            # Optional: print status periodically
            if int(timestamp_ms) % 5000 == 0:
                print(f"Logging... {timestamp_ms/1000:.1f}s elapsed")
                
    except KeyboardInterrupt:
        robot.go_limp()
        print("\n\nLogging stopped.")
        print(f"Data saved to: {csv_filename}")
