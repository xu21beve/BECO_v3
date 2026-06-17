###############################################################################
# beco_v2.py
# Python conversion of beco_v2.cpp / beco_v2.h
#
# Structural changes from C++:
#   1. Header/source split collapsed into one file — Python has no headers.
#   2. Arduino globals (millis, delay, Serial) replaced with time module calls:
#        millis()  -> time.time() * 1000  (returns float ms)
#        delay(ms) -> time.sleep(ms / 1000)
#        Serial.begin() / Serial.println() -> print()
#   3. LSS bus init uses lss.initBus() / lss.closeBus() module functions
#      (matching lss.py's module-level API) instead of LSS::initBus().
#   4. 'new LSS(ID)' -> LSS(ID) — no heap allocation syntax in Python.
#   5. Module constructor overloading replaced with a single __init__ that
#      accepts an optional parent_module keyword argument (Python has no
#      function overloading).
#   6. std::vector<> -> Python list.
#   7. std::cout / std::printf -> print().
#   8. math.floor() used instead of C floor().
#   9. The first-module check in move_module is now:
#        "if self.parent_module is None" — same logic, Pythonic form.
###############################################################################

import math
import time
# import lib.lss                          # lss.py — provides LSS class and initBus/closeBus
import lib.lss.lss as lss
import lib.lss.lss_const as lssc            # required by lss.py internally; imported for completeness

LSS_BAUD   = 115200                 # matches LSS_DefaultBaud convention
LSS_SERIAL = "COM10"        # replace with the actual serial port name


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

class Motor:
    def __init__(self, motor_id: int, amplitude: float, period: float,
                 servo_angle_offset: float, phase_offset_frac: float):
        self.ID                 = motor_id
        self.amplitude          = amplitude
        self.period             = period
        self.servo_angle_offset = servo_angle_offset
        self.phase_offset_frac  = phase_offset_frac
        self.motor_obj          = lss.LSS(motor_id)   # was: new LSS(ID)
        self.angle_pos          = 0.0

    def triangle_wave(self, t_ms: float) -> float:
        """Return the target angle for time t_ms (milliseconds)."""
        phase_time = (t_ms - self.phase_offset_frac * self.period) / self.period
        phase_time = phase_time - math.floor(phase_time)

        if phase_time <= 0.5:
            # Downwards slope
            out = -4 * self.amplitude * phase_time + self.amplitude + self.servo_angle_offset
        else:
            out = 4 * self.amplitude * phase_time - 3 * self.amplitude + self.servo_angle_offset

        return out

    def move(self, angle_pos: float):
        self.motor_obj.move(angle_pos)

    def print_current(self):
        print(f"Motor with ID {self.ID} current draw (amps): {self.motor_obj.getCurrent()}")

    def get_current(self) -> float:
        return self.motor_obj.getCurrent()


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class Module:
    def __init__(self, front_leg: Motor, back_leg: Motor, body: Motor,
                 fb_offset: float, fr_offset: float,
                 f_offset: float = 0.0,
                 parent_module: "Module | None" = None):
        """
        Two construction modes from C++ merged into one __init__:

          First module  — pass f_offset explicitly, leave parent_module=None.
          Child module  — pass parent_module; f_offset is derived automatically
                          (mirrors the second C++ constructor).
        """
        self.front_leg = front_leg
        self.back_leg  = back_leg
        self.body      = body
        self.fb_offset = fb_offset
        self.fr_offset = fr_offset
        self.parent_module = parent_module

        if parent_module is not None:
            # Derive f_offset from parent, same as second C++ constructor
            self.f_offset = parent_module.get_f_offset() + parent_module.get_fr_offset()
        else:
            self.f_offset = f_offset

    def move_module(self, time_ms: float):
        # First module (no parent) moves its front leg; child modules skip it
        # (their front leg position is driven by the parent's back leg timing)
        if self.parent_module is None:
            self.front_leg.move(self.front_leg.triangle_wave(time_ms))
            print(f"front leg target pose: {self.front_leg.triangle_wave(time_ms)}")
        self.back_leg.move(self.back_leg.triangle_wave(time_ms))
        self.body.move(self.body.triangle_wave(time_ms))

    def get_motor_currents(self) -> list[float]:
        return [
            self.front_leg.get_current(),
            self.back_leg.get_current(),
            self.body.get_current(),
        ]

    def print_motor_currents(self):
        print(f"-------- Module {self.get_module_id()} --------")
        self.front_leg.print_current()
        self.back_leg.print_current()
        self.body.print_current()

    def get_module_id(self) -> int:
        """Walk the parent chain to determine this module's index."""
        module_id   = 0
        new_parent = self.parent_module
        while new_parent is not None:
            new_parent = new_parent.get_parent_module()
            module_id += 1
        return module_id

    def get_parent_module(self) -> "Module | None":
        return self.parent_module

    def get_fb_offset(self) -> float:
        return self.fb_offset

    def get_fr_offset(self) -> float:
        return self.fr_offset

    def get_f_offset(self) -> float:
        return self.f_offset


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

class Robot:
    def __init__(self, modules: list[Module]):
        self.modules    = modules
        self.has_setup  = False
        self.start_time = 0.0          # will be set in setup(), stored as ms

    def setup(self):
        # Initialise the LSS bus (replaces LSS::initBus + Serial.begin)
        lss.initBus(LSS_SERIAL, LSS_BAUD)
        time.sleep(2)                  # delay(2000)
        print("LSS bus initialised.")

        for mod in self.modules:
            mod.move_module(self._millis())

        time.sleep(15)                 # delay(15000) — initial settle
        print("Done setting up.")
        self.has_setup  = True
        self.start_time = self._millis()

    def loop(self):
        if not self.has_setup:
            print("Need to run setup first before looping. Exiting loop.")
            return

        elapsed = self._millis() - self.start_time

        for mod in self.modules:
            mod.move_module(elapsed)

        time.sleep(0.1)                # pause sending commands for 20 ms

    def print_motor_currents(self):
        for mod in self.modules:
            mod.print_motor_currents()

    def get_motor_currents(self) -> list[list[float]]:
        return [mod.get_motor_currents() for mod in self.modules]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _millis() -> float:
        """Return current time in milliseconds (replaces Arduino millis())."""
        return time.time() * 1000.0