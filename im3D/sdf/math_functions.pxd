#=== 32-bit versions ===========================================================
cdef extern from "math.h":
    float sqrtf(float) nogil
    float fabsf(float) nogil

cdef inline float f_max(float a, float b) nogil:
    return a if a >= b else b

cdef inline float f_min(float a, float b) nogil:
    return a if a <= b else b

cdef inline float f_sign(float a) nogil:
    return 0.0 if a == 0.0 else sqrt(a*a)/a

#=== 64-bit versions ===========================================================
cdef extern from "math.h":
    double sqrt(double) nogil
    double fabs(double) nogil

cdef inline double d_max(double a, double b) nogil:
    return a if a >= b else b

cdef inline double d_min(double a, double b) nogil:
    return a if a <= b else b

cdef inline double d_sign(double a) nogil:
    return 0.0 if a == 0.0 else sqrt(a*a)/a
