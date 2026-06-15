#include "beco_v2.h"
#include <iostream>
// This class allows you to easily add on modules to a longer robot

/** Classes
- Full robot
- Module (two legs and one body)
- Motor (which leg and body are both objects of inherit)
*/

/** Attributes/Methods
  // Full robot
    - Start time
    - Period
    - Standard leg amplitude
    - Front leg standard phase offset
    - Servo deg scale
    - Angle offsets between leg and body motors, FB and FR

  // Motor
    - Motor ID
    - Motor object
    - Current position
    - Amplitude
    - Triangle_wave
    - Move

  // Module
    - 2 legs
    - 1 body 
    - Whether it is the first module
    - Front leg standard phase offset
    - Function to move whole module
*/

// Robot method implementations
void Robot::setup() {
  // Initialize the LSS bus
  LSS::initBus(LSS_SERIAL, LSS_BAUD);
  // Wait for the LSS to boot
  delay(2000);
  Serial.begin(115200);

  for(Module *mod : modules)
  {
    mod->move_module(start_time);
  }
  
  delay(15000);
  Serial.println("Done setting up.");
  has_setup = true;
  start_time = millis();
}

void Robot::loop() {
  if (!has_setup) {
    std::printf("Need to run setup first before looping. Exiting loop.");
    return;
  }

  unsigned long time = (millis()-start_time);

  for(Module *mod : modules) {
    mod->move_module(time);
  }
  
  delay(100);
}

void Robot::print_motor_currents() {
  for (Module *mod : modules) {
    mod->print_motor_currents();
  }
}

std::vector<std::vector<float>> Robot::get_motor_currents() {
  std::vector<std::vector<float>> motor_currents;
  for (Module *mod : modules) {
    motor_currents.push_back(mod->get_motor_currents());
  }
  return motor_currents;
}