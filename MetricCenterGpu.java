//******************************************************************************
//
// File:    MetricCenterGpu.java
// Using Package: edu.rit.pj2
//
// This Java source file is copyright (C) 2015 by Utkarsh Bhatia. All rights
// reserved. For further information, contact the author, Utkarsh Bhatia, at
// uxb9472@rit.edu.
//
// This class is extending Task class as given in the PJ2 parallel java library
// made by Professor Alan Kaminsky, the given reference to parallel java library and its sample
// code can be referenced from http://www.cs.rit.edu/~ark/bcbd/#source and http://www.cs.rit.edu/~ark/pj2.shtml
// This class is used for running the program with MetricCenterGpu.cubin for running the
// program in CUDA which works in multiple GPU cores.
//
// Details for PJ2 library as available on http://www.cs.rit.edu/~ark/pj2.shtml
// The library has been made available to General Public under GPL license by 
// Professor Alan Kaminsky. The copyright (C) 2015 to pj2 library is held by Alan Kaminsky.
// PJ2 is free software; you can redistribute it and/or modify it under the terms of
// the GNU General Public License as published by the Free Software Foundation;
// either version 3 of the License, or (at your option) any later version.
//
// PJ2 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;

import edu.rit.gpu.CacheConfig;
import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuStructArray;
import edu.rit.gpu.Kernel;
import edu.rit.gpu.Module;
import edu.rit.gpu.Struct;
import edu.rit.pj2.Task;

/**
 * Class MetricCenterGpu is a GPU parallel program to compute the metric center from
 * a given set of cordinates. This is a metric center finding problem.
 * <P>
 * <BR><TT><I>filePath</I></TT> = path for the input file
 * 								containing the cordinates
 *
 * @author  Utkarsh Bhatia
 * @version 06-Nov-2015
 */
public class MetricCenterGpu extends Task {

	//Declaring global variables for list and storing final length of the input set of cordinates
	int totalLength;
	ArrayList<VectorInput> cordList;
	GpuStructArray<VectorInput> pList;
	GpuStructArray<VectorResult> rList;

	/**
	 * @author Utkarsh
	 * 
	 * Kernel function interface
	 *
	 */
	private static interface MetricCenterKernel extends Kernel {
		public void metricCalculate(GpuStructArray<VectorInput> pList, GpuStructArray<VectorResult> rList, int N);
	}

	/**
	 * Task main program.
	 */
	public void main(String[] args) throws Exception {	

		//Buffered Reader for reading the file
		BufferedReader buffReader = null;

		try {
			//Parsing command line arguments
			if (args.length != 1)
				throw new IllegalArgumentException();
			
			//set the dynamic array list to read the input file
			cordList = new ArrayList<VectorInput>();
			String[] tempArray = null;
			double x, y;
			VectorInput tempCord;
			//start reading the input file via buffered reader
			buffReader = new BufferedReader(new FileReader(args[0]));
			String readLine = buffReader.readLine();
			//reading the file line by line
			while (readLine != null) {
				tempArray = readLine.split(" ");
				x = Double.parseDouble(tempArray[0]);
				y = Double.parseDouble(tempArray[1]);
				tempCord = new VectorInput(x, y);
				cordList.add(tempCord);
				readLine = buffReader.readLine();
			}

			// Initialize GPU.
			Gpu gpu = Gpu.gpu();
			gpu.ensureComputeCapability(2, 0);
			int mpCount = gpu.getMultiprocessorCount();

			// Set up GPU variables.
			Module module = gpu.getModule("MetricCenterGpu.cubin");

			//declaring the point list and the result list
			pList = gpu.getStructArray(VectorInput.class, cordList.size());
			rList = gpu.getStructArray(VectorResult.class, mpCount);

			//intialize the point list for sending to the gpu kernel
			for (int i = 0; i < cordList.size(); i++) {
				pList.item[i] = new VectorInput(cordList.get(i).getX(),
						cordList.get(i).getY());
			}

			//initialize the result list for sending to the gpu kernel
			for (int i = 0; i < mpCount; i++) {
				rList.item[i] = new VectorResult(-1, 0);
			}

			// Set up GPU kernel.
			MetricCenterKernel kernel = module.getKernel(MetricCenterKernel.class);
			kernel.setBlockDim(1024);
			kernel.setGridDim(mpCount);
			kernel.setCacheConfig(CacheConfig.CU_FUNC_CACHE_PREFER_L1);

			//send the point list and the initialized result list to the gpu kernel
			pList.hostToDev();
			rList.hostToDev();
			kernel.metricCalculate(pList, rList, pList.length());

			//get the point list and the result list for the gpu
			pList.devToHost();
			rList.devToHost();

			//final reduction for the result list received from the gpu
			//resultReduction(rList);
			VectorResult.reduceResult(rList, pList);
		
			} catch(IllegalArgumentException ie){ //catch Illegal arguments
				System.err.println("Input Arguments not verified. Please give the correct input path for file.");
				System.err.println ("Usage: java MetricCenter running with pj2. Arguments taken: <filePath>");
				System.err.println ("<filePath> = Path for input file containing cordinates");
				System.err.println ("Remember: file path should be entered correctly");
				System.err.println ("Please check and enter correct arguments. Exception: " + ie);
			} catch(FileNotFoundException fe){ //catch if the file not present in the given location
				System.err.println("File Not Found. Please give the correct input path for file.");
				System.err.println ("Usage: java MetricCenter running with pj2. Arguments taken: <filePath>");
				System.err.println ("<filePath> = Path for input file containing cordinates");
				System.err.println ("Remember: file path should be entered correctly");
				System.err.println ("Please check and enter correct arguments. Exception: " + fe);
			} catch (Exception e) { //catch other exception
				System.err.println ("Usage: java MetricCenter running with pj2. Arguments taken: <filePath>");
				System.err.println ("<filePath> = Path for input file containing cordinates");
				System.err.println ("Remember: file path should be entered correctly");
				System.err.println ("Please check and enter correct arguments. Exception: " + e);
			}
		
		}

	/**
	 * Specify that this task requires one core.
	 */
	protected static int coresRequired() {
		return 1;
	}

	/**
	 * Specify that this task requires one GPU accelerator.
	 */
	protected static int gpusRequired() {
		return 1;
	}

	/**
	 * @author Utkarsh
	 *			Structure for a 2-D vector.
	 */
	private static class VectorInput extends Struct {
		//class variables for x and y cordinate
		public double x;
		public double y;

		// Construct a new vector.
		public VectorInput(double x, double y) {
			this.x = x;
			this.y = y;
		}

		// Returns the size in bytes of the C struct.
		public static long sizeof() {
			return 16;
		}

		// Write this Java object to the given byte buffer as a C struct.
		public void toStruct(ByteBuffer buf) {
			buf.putDouble(x);
			buf.putDouble(y);
		}

		// Read this Java object from the given byte buffer as a C struct.
		public void fromStruct(ByteBuffer buf) {
			x = buf.getDouble();
			y = buf.getDouble();
		}

		public double getX() {
			return x;
		}

		public void setX(double x) {
			this.x = x;
		}

		public double getY() {
			return y;
		}

		public void setY(double y) {
			this.y = y;
		}
	}
	
	/**
	 * @author Utkarsh
	 *			Structure for a return vector.
	 */
	private static class VectorResult extends Struct {
		//class variables for metric center radius and corresponding index
		public double r;
		public int i;

		// Construct a new vector.
		/**
		 * @param r
		 * @param i
		 */
		public VectorResult(double r, int i) {
			this.r = r;
			this.i = i;
		}

		// Returns the size in bytes of the C struct.
		/**
		 * @return
		 */
		public static long sizeof() {
			return 16;
		}

		// Write this Java object to the given byte buffer as a C struct.
		public void toStruct(ByteBuffer buf) {
			buf.putDouble(r);
			buf.putInt(i);
		}

		// Read this Java object from the given byte buffer as a C struct.
		public void fromStruct(ByteBuffer buf) {
			r = buf.getDouble();
			i = buf.getInt();
		}
		
		/**
		 * @param rList
		 * 			result list received from gpu kernel
		 * @param pList
		 * 			
		 */
		public static void reduceResult(GpuStructArray<VectorResult> rList, GpuStructArray<VectorInput> pList){
			double ansRadius; int ansIndex;
			ansIndex = rList.item[0].i; ansRadius = rList.item[0].r;
			for (int i = 1; i < rList.length(); i++) {
				if(ansRadius > rList.item[i].r){
					ansRadius = rList.item[i].r;
					ansIndex = rList.item[i].i;
				}
			}
			//print the final output
			printOutput(ansRadius, ansIndex, pList);
		}

		/**
		 * @param ansRadius
		 * 				final radius obtained after reduction
		 * @param ansIndex
		 * 				final index obtained after reduction
		 * @param pList
		 * 			point list as taken from the input file
		 */
		private static void printOutput(double ansRadius, int ansIndex, GpuStructArray<VectorInput> pList) {
			System.out.print ((int)ansIndex +" (");
			System.out.printf ("%.5g", pList.item[ansIndex].x);
			System.out.print(",");
			System.out.printf ("%.5g", pList.item[ansIndex].y);
			System.out.println(")");
			System.out.printf ("%.5g", ansRadius);
			System.out.println();
		}
	}

}

/**
 * @author Utkarsh Bhatia
 * 
 * external class for storing cordinates from the
 * given input file
 *
 */
class Cordinate{
	//class variables for x and y cordinates
	double xC;
	double yC;
	
	/**
	 * constructor for cordinate
	 * @param xC
	 * @param yC
	 */
	public Cordinate(double xC, double yC) {
		super();
		this.xC = xC;
		this.yC = yC;
	}

	/**
	 * @return
	 * 		x cordinate
	 */
	public double getxC() {
		return xC;
	}

	/**
	 * @param xC
	 * 		set x cordinate
	 */
	public void setxC(double xC) {
		this.xC = xC;
	}

	/**
	 * @return
	 * 		y cordinate
	 */
	public double getyC() {
		return yC;
	}

	/**
	 * @param yC
	 * 		set y cordinate
	 */
	public void setyC(double yC) {
		this.yC = yC;
	}
	
}