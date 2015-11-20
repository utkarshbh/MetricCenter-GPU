Consider a list of two-dimensional points with coordinates (x, y). Suppose we pick one of the points, and we draw a circle centered at that point that just barely encloses all the points. Consider drawing such a circle for each of the points in the list. We want to find the point that yields the circle with the smallest area. This point is called the metric center of the list of points. 

The parallel program is designed to run on a CUDA capable GPU.

Program Input and Output

The program's command line argument is the name of an input file. Each line of the input file must consist of two double precision floating point numbers (type double) separated by whitespace, that specify the X and Y coordinates of one 2-D point. The points are indexed starting at 0; that is, the first line is the point at index 0, the second line is the point at index 1, the third line is the point at index 2, and so on. The input file must contain at least two points.

The program's first output line gives the metric center of the list of points. The second output line gives the radius of the metric center's circle, a double precision floating point number. 

If more than one point yields a circle with the same smallest area, the metric center is the point with the smaller index.

To run:
- Compile the Java main program using this command: 

      $ javac MetricCenterGpu.java

- Compile the CUDA kernel using this command:

      $ nvcc -cubin -arch compute_20 -code sm_20 --ptxas-options="-v" -o MetricCenterGpu.cubin MetricCenterGpu.cu
    
- Run the program using this command (substituting the proper command line argument):

      $ java pj2 MetricCenterGpu <file>
