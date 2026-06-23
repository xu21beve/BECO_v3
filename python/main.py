###############################################################################
# main.py
# Python conversion of main.cpp (Arduino sketch)
#
# Structural changes from C++:
#   1. Arduino setup()/loop() paradigm replaced with a plain Python
#      if __name__ == "__main__" entry point and an explicit while True loop.
#   2. C++ #define constants -> Python module-level constants.
#   3. ModuleConfig struct -> Python dataclass (same fields, same values).
#   4. 'new Motor(...) / new Module(...) / new Robot(...)' -> direct
#      constructor calls — Python has no heap allocation syntax.
#   5. Robot({module}) (C++ initializer list) -> Robot([module]) (Python list).
###############################################################################

from dataclasses import dataclass
from beco_v3 import Motor, Module, Robot
import time

# --- LSS motor IDs ----------------------------------------------------------

LSS_ID_LEG_1  = 0
LSS_ID_LEG_2  = 2
LSS_ID_BODY_1 = 1

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
    period                  = 10000,   # originally 3000
    body_amp                = 600,    # servo scale (not degrees); default ~700; previously 600 (set to 0 for testing legs without body)
    phi_FB                  = 0.7,    # value in [0, 1)
    phi_FR                  = 0.3,    # value in [0, 1)
    frontleg_standard_phase = 0,
    servo_deg_scale         = 10,
    leg_amp                 = 1600,    # servo scale; 900 ≈ fully extended
    # angle_offset_leg        = -900,   # = -leg_amp
    angle_offset_leg        = 0,     # TODO: add independent leg angle offsets
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
                f_offset=0.0)           # first module: no parent

robot = Robot([module])

# --- Entry point (replaces Arduino setup() + loop()) -----------------------

if __name__ == "__main__":
    robot.setup()

    while True:
        robot.loop()
        # robot.print_motor_currents()
        # To retrieve an n×3 matrix of currents instead:
        #   currents = robot.get_motor_currents()