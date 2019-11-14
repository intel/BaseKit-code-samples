//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <math.h>
#include <vector>

using namespace std;
//Projectile class
class Projectile {
    private:
        float m_angle;
        float m_velocity;
        float m_range;
        float m_totalTime;
        float m_maxHeight;
    public:
		Projectile()
		{
			m_angle = 0;
			m_velocity =0;
			m_range = 0;
			m_totalTime = 0;
			m_maxHeight = 0;
		}

		Projectile(float angles, float velocitys,float r, float t,float m)
		{                     
			m_angle = angles;
			m_velocity = velocitys;
			m_range = r;
			m_totalTime = t;
			m_maxHeight = m;
		}

		float getangle() const{
			return m_angle;
		}
		float getvelocity() const{
			return m_velocity;
		}
		//Set the Range and total flight time
		void setRangeandTime(float frange,float ttime,float angle_s, float velocity_s, float height_s) {
			m_range = frange;
			m_totalTime = ttime;
			m_angle = angle_s;
			m_velocity = velocity_s;
			m_maxHeight = height_s;
		}
		float getRange() const{
			return m_range;
		}

		float gettotalTime() const{
			return m_totalTime;
		}

		float getmaxHeight() const {
			return m_maxHeight;
		}
		//Overloaded == operator to compare two projectile objects
		friend bool operator==(const Projectile &a, const Projectile &b)
		{
			return (a.m_angle == b.m_angle) && (a.m_velocity == b.m_velocity) && (a.m_range == b.m_range) && (a.m_totalTime == b.m_totalTime) && (a.m_maxHeight == b.m_maxHeight);
		}
		//Ostream operator overloaded to display a projectile object
		friend ostream& operator << (ostream& out, const Projectile& obj){
			out<<"Angle: "<<obj.getangle()<<" Velocity: "<<obj.getvelocity()<<" Range: "<<obj.getRange()<<" Total time: "<<obj.gettotalTime()<<" Maximum Height: "<<obj.getmaxHeight()<<"\n";
			return out;
		}
};