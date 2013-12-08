#ifndef CAMERA_H
#define CAMERA_H

#include "Vec3.h"


class GL_Camera
{
 private:
	
  float dist;

  Vec3<float> point;
  Vec3<float> upV;
  Vec3<float> rightV;
  Vec3<float> eyeV;

  /* View point with vectors defining the local axis
   *
   *              upV
   *               |   rightV
   *               |  /
   *               | /
   *               |/
   *               * point
   *                \
   *                 \
   *                  \
   *                   eyeV
   */

 public:
  
 GL_Camera() : dist(1), point(0,0,0), rightV(1,0,0), upV(0,1,0), eyeV(0,0,1) {}
  
  // Note: You should call glLoadIdentity before using Render
  inline void render()
  {
    Vec3<float> eye = point + eyeV * dist;

    gluLookAt( eye.x, eye.y, eye.z,
	       point.x, point.y, point.z,
	       upV.x, upV.y, upV.z );
  }
  
  // Change the point we are viewing
  inline void move( const Vec3<float>& moveV )
  {
    point += moveV;
  }

  // Change the point we are viewing with respect to the local axes
  inline void pan( const Vec3<float>& panV )
  {
    point += rightV * panV.x;
    point += upV    * panV.y;
    point += eyeV   * panV.z;
  }

  // Set a new view point
  inline void setViewPoint( const Vec3<float>& point_ )
  {
    point = point_;
  }

  // Rotate the view about the local x-axis
  inline void rotateX( float angle )
  {
    // Rotate eye vector about the right vector
    eyeV = eyeV * cosf(angle) + upV * sinf(angle);
    eyeV.normalize();    

    // Compute the new up vector by cross product
    upV = eyeV.cross(rightV);
    upV.normalize();
  }

  // Rotate the view about the local y-axis
  inline void rotateY( float angle )
  {
    // Rotate eye about the up vector
    eyeV = eyeV * cosf(angle) + rightV * sinf(angle);
    eyeV.normalize();

    // Compute the new right vector by cross product
    rightV = upV.cross(eyeV);
    rightV.normalize();
  }

  // Zoom by a scale factor
  inline void zoom( float d )
  {
    dist *= d;
  }
};



#endif
