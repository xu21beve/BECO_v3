#include <Arduino.h>
#include "beco_v2.h"

// ID set to default LSS ID = 0
#define LSS_ID_LEG_1		(1)
#define LSS_ID_LEG_2		(0)
#define LSS_ID_BODY_1		(3)

// Module configuration struct
struct ModuleConfig {
  unsigned int period;
  float body_amp;
  float phi_FB;
  float phi_FR;
  float frontleg_standard_phase;
  float servo_deg_scale;
  float leg_amp;
  float angle_offset_leg;
  float angle_offset_body;
};

// Module 1 configuration
ModuleConfig module1 = {
  .period = 3000,
  .body_amp = 600,              // in servo scale, not in degree, "default": 700
  .phi_FB = 0.9,                // phase difference between front leg and body movement, value between [0,1)
  .phi_FR = 0.3,                // phase difference between front and rear leg, value between [0,1)
  .frontleg_standard_phase = 0,
  .servo_deg_scale = 10,
  .leg_amp = 900,               // in servo scale, half of 180 is a whole round, from min to max (900 is fully extended?)
  .angle_offset_leg = -900,      // -leg_amp
  .angle_offset_body = 0         // body motor abs angle offset
};

// Create motors for first module
Motor *front_leg = new Motor(LSS_ID_LEG_1, module1.leg_amp, module1.period, module1.frontleg_standard_phase, module1.angle_offset_leg);
Motor *back_leg = new Motor(LSS_ID_LEG_2, module1.leg_amp, module1.period, module1.phi_FR, module1.angle_offset_leg);
Motor *body = new Motor(LSS_ID_BODY_1, module1.body_amp, module1.period, module1.phi_FB, module1.angle_offset_body);

// Create first module
Module *module = new Module(front_leg, back_leg, body, module1.phi_FB, module1.phi_FR, 0.0);  // f_offset = 0.0

// Create robot with one module
Robot *robot = new Robot({module});

void setup() {
  // Initialize and start the robot
  robot->setup();
}

void loop() {
  // Run the robot loop
  robot->loop();
  robot->print_motor_currents(); // Can also retrieve nx3 matrix (n is number of total motors) of motor currents using get_motor_currents()
}

// Robot will move continuously based on the triangle wave functions
