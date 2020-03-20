//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <math.h>
#include <iostream>
#include <vector>

using namespace std;
// Projectile class
class Projectile {
 private:
  float m_angle_;
  float m_velocity_;
  float m_range_;
  float m_totalTime_;
  float m_maxHeight_;

 public:
  Projectile() {
    m_angle_ = 0;
    m_velocity_ = 0;
    m_range_ = 0;
    m_totalTime_ = 0;
    m_maxHeight_ = 0;
  }

  Projectile(float angle, float velocity, float range, float time,
             float maxheight) {
    m_angle_ = angle;
    m_velocity_ = velocity;
    m_range_ = range;
    m_totalTime_ = time;
    m_maxHeight_ = maxheight;
  }

  float getangle() const { return m_angle_; }
  float getvelocity() const { return m_velocity_; }
  // Set the Range and total flight time
  void setRangeandTime(float frange, float ttime, float angle_s,
                       float velocity_s, float height_s) {
    m_range_ = frange;
    m_totalTime_ = ttime;
    m_angle_ = angle_s;
    m_velocity_ = velocity_s;
    m_maxHeight_ = height_s;
  }
  float getRange() const { return m_range_; }

  float gettotalTime() const { return m_totalTime_; }

  float getmaxHeight() const { return m_maxHeight_; }
  // Overloaded == operator to compare two projectile objects
  friend bool operator!=(const Projectile& a, const Projectile& b) {
    return (a.m_angle_ != b.m_angle_) || (a.m_velocity_ != b.m_velocity_) ||
           (a.m_range_ != b.m_range_) || (a.m_totalTime_ != b.m_totalTime_) ||
           (a.m_maxHeight_ != b.m_maxHeight_);
  }
  // Ostream operator overloaded to display a projectile object
  friend ostream& operator<<(ostream& out, const Projectile& obj) {
    out << "Angle: " << obj.getangle() << " Velocity: " << obj.getvelocity()
        << " Range: " << obj.getRange() << " Total time: " << obj.gettotalTime()
        << " Maximum Height: " << obj.getmaxHeight() << "\n";
    return out;
  }
};
