#ifndef BECO_V2_H
#define BECO_V2_H

#include "LSS.h"
#include <vector>

#define LSS_BAUD	(LSS_DefaultBaud)
#define LSS_SERIAL	(Serial)

class Motor {
  private: 
    int ID;
    LSS *motor_obj;
    float angle_pos, amplitude, period, servo_angle_offset, phase_offset_frac;
  
  public:
  Motor(int ID, float amplitude, float period, float servo_angle_offset, float phase_offset_frac) {
    this->ID = ID;
    this->amplitude = amplitude;
    this->period = period;
    this->servo_angle_offset = servo_angle_offset;
    this->phase_offset_frac = phase_offset_frac;
    this->motor_obj = new LSS(ID);
  }

  float triangle_wave(float t_ms)
  {
    float out;
    float phase_time = (t_ms - phase_offset_frac * period) / period;
    phase_time = phase_time - floor(phase_time);
    
    if(phase_time <= 0.5)
    {
      out = -4*amplitude*(phase_time) + amplitude + servo_angle_offset; // Downwards slope, *4 because amplitude is half of delta y and period is double delta x
    }
    else
    {
      out = 4*amplitude*(phase_time) -3*amplitude + servo_angle_offset;
    }
    return out;
  }

  void move(float angle_pos) {
    motor_obj->move(angle_pos);
  }

  void print_current() {
    std::cout << "Motor with ID " + std::to_string(ID) + " current draw (amps): " + std::to_string(motor_obj->getCurrent()) << std::endl;
  }

  float get_current() {
    return motor_obj->getCurrent();
  }
};

class Module {
  private: 
  Motor *front_leg{nullptr}, *back_leg{nullptr}, *body{nullptr};
  float fb_offset, fr_offset, f_offset;
  Module *parent_module{nullptr};

  public:
  Module(Motor *front_leg, Motor *back_leg, Motor *body, float fb_offset, float fr_offset, float f_offset){
    this->front_leg = front_leg;
    this->back_leg = back_leg;
    this->f_offset = f_offset;
    this->body = body;
  }
  
  Module(Motor *front_leg, Motor *back_leg, Motor *body, float fb_offset, float fr_offset, Module *parent_module){
    this->front_leg = front_leg;
    this->back_leg = back_leg;
    this->body = body;
    this->parent_module = parent_module;
    this->f_offset = parent_module->get_f_offset() + parent_module->get_fr_offset();
  }

  void move_module(float time) {
    if(parent_module == nullptr)
    {
      front_leg->move(front_leg->triangle_wave(time));
    }
    back_leg->move(back_leg->triangle_wave(time));
    body->move(body->triangle_wave(time));
  }

  std::vector<float> get_motor_currents() {
    return std::vector<float>{front_leg->get_current(), back_leg->get_current(), body->get_current()};
  }

  void print_motor_currents() {
    std::cout << "-------- Module " + std::to_string(get_module_id()) + " --------" << std::endl;
    front_leg->print_current();
    back_leg->print_current();
    body->print_current();
  }

  int get_module_id() {
    int id = 0;
    Module *new_parent = parent_module;
    
    while (new_parent != nullptr) {
        new_parent = new_parent->get_parent_module();
        id++;
    }

    return id;
  }

  Module *get_parent_module () {
    return parent_module;
  }
  
  float get_fb_offset() const { return fb_offset; }
  float get_fr_offset() const { return fr_offset; }
  float get_f_offset() const { return f_offset; }
};

class Robot {
  private:
  std::vector<Module *> modules;
  bool has_setup{false};
  float start_time{0.0};

  public:
  Robot(std::vector<Module *> modules){
    this->modules = modules;
  }

  void setup();
  void loop();
  void print_motor_currents();
  std::vector<std::vector<float>> get_motor_currents();
};

#endif
