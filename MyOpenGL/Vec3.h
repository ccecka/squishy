#ifndef VEC3_H
#define VEC3_H

template <typename T>
class Vec3
{
 public:

  T x, y, z;

  // Constructors
  Vec3(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) {}
  template <typename D>
  Vec3(const Vec3<D>& v) : x(v.x), y(v.y), z(v.z) {}

  // Operator Overloads
  template <typename D>
  inline bool operator==(const Vec3<D>& v) const {
    return (x == v.x && y == v.y && z == v.z);
  }

  template <typename D>
  inline Vec3<T>& operator=(const Vec3<D>& v) {
    x = v.x; y = v.y; z = v.z;
    return *this;
  }

  template <typename D>
  inline Vec3<T> operator+(const Vec3<D>& v) const {
    return Vec3<T>(x + v.x, y + v.y, z + v.z);
  }

  template <typename D>
  inline Vec3<T> operator+(D S) const {
    return Vec3<T>(x + S, y + S, z + S);
  }

  template <typename D>
  inline void operator+=(const Vec3<D>& v) {
    x += v.x; y += v.y; z += v.z;
  }

  template <typename D>
  inline void operator+=(D S) {
    x += S; y += S; z += S;
  }

  inline Vec3<T> operator-() const {
    return Vec3<T>(-x, -y, -z);
  }

  template <typename D>
  inline Vec3<T> operator-(const Vec3<D>& v) const {
    return Vec3<T>(x - v.x,  y - v.y,  z - v.z);
  }

  template <typename D>
  inline Vec3<T> operator-(D S) const {
    return Vec3<T>(x - S, y - S, z - S);
  }

  template <typename D>
  inline void operator-=(const Vec3<D>& v) {
    x -= v.x; y -= v.y; z -= v.z;
  }

  template <typename D>
  inline void operator-=(D S) {
    x -= S; y -= S; z -= S;
  }

  template <typename D>
  inline Vec3<T> operator/(const Vec3<D>& v) const {
    return Vec3<T>(x / v.x,  y / v.y,  z / v.z);
  }

  template <typename D>
  inline Vec3<T> operator/(D S) const {
    T s = 1.0 / S;
    return Vec3<T>(x * s , y * s, z * s);
  }

  template <typename D>
  inline void operator/=(const Vec3<D>& v) {
    x /= v.x; y /= v.y; z /= v.z;
  }

  template <typename D>
  inline void operator/=(D S) {
    x /= S; y /= S; z /= S;
  }

  template <typename D>
  inline Vec3<T> operator*(const Vec3<D>& v) const {
    return Vec3<T>(x * v.x,  y * v.y,  z * v.z);
  }

  template <typename D>
  inline Vec3<T> operator*(D S) const {
    return Vec3<T>(x * S,  y * S,  z * S);
  }

  template <typename D>
  inline void operator*=(const Vec3<D>& v) {
    x *= v.x; y *= v.y; z *= v.z;
  }

  template <typename D>
  inline void operator*=(D S) {
    x *= S; y *= S; z *= S;
  }

  // Functions
  template <typename D>
  inline T dot(const Vec3<D>& v) const {
    return v.x*x + v.y*y + v.z*z;
  }

  template <typename D>
  inline Vec3<T> cross(const Vec3<D>& v) const {
    return Vec3<T>(y*v.z - z*v.y,
                   z*v.x - x*v.z,
                   x*v.y - y*v.x);
  }

  inline T mag() const {
    return sqrt( x*x + y*y + z*z );
  }

  inline T magSq() const {
    return x*x + y*y + z*z;
  }

  template <typename D>
  inline T distSq(const Vec3<D>& v) const {
    T a = x - v.x;
    T b = y - v.y;
    T c = z - v.z;
    return a*a + b*b + c*c;
  }

  template <typename D>
  inline T dist(const Vec3<D>& v) const {
    return sqrt( distSq(v) );
  }

  inline void normalize() {
    T mag = ( x*x + y*y + z*z );
    if (mag == 0) { return; }

    T s = 1.0/sqrt(mag);
    x *= s; y *= s; z *= s;
  }

  friend ostream& operator<<(ostream& os, const Vec3<T>& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  }
};

#endif
