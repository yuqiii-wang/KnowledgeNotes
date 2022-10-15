# RGB-D Camera

Besides image pixel mapping, camera can measure pixel's depth.

There are two solutions:
* *structured infrared light*

Camera calculates the distance between the object and itself based on the returned structured light pattern.

* *time-of-flight* (ToF)

Camera emits a light pulse to the target
and then determines the distance according to the beam’s time of flight.