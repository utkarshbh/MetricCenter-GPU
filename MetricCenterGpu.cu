//******************************************************************************
//
// File:    MetricCenterGpu.cu
// Unit:    MetricCenterGpu kernel function
//
// This C/CUDA source file is copyright (C) 2014 by Utkarsh Bhatia. 
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
// Created on: Nov 6, 2015
// 
// Author: Utkarsh Bhatia
// 
//******************************************************************************

// Number of threads per block.
#define NT 1024

// Structure for a 2-D vector.
typedef struct
{
 double x;
 double y;
}
VectorInput;

// Structure for a radius and index vector
typedef struct
{
 double r;
 int i;
}
VectorResult;

// Per-thread variables in shared memory.
__shared__ VectorResult resultArray[NT];

/*Calculate temperory distance for the current point being processed in the current thread
 *
 * @param  cord1 cord1 to calculate distance
 * @param  cord2 cord2 to calculate distance
 *
 * @return
 * 	returns distance from cord1 to cord2
 */
__device__ double tempDistance(VectorInput *cord1, VectorInput *cord2)
{
	double diffX = cord1->x - cord2->x;
	double diffY = cord1->y - cord2->y;
	double tempDist = sqrt( pow(diffX, 2) + pow(diffY, 2) );
	return tempDist;
}

/*get maximum distance for current selected point at block level
 *
 * @param  oldMax current maximum value from the selected point
 * @param  newDist calculated new distance from the selected point
 *
 * @return
 * 	returns new maximum distance
 */
__device__ double getMaxDist(double oldMax, double newDist)
{
	if(newDist > oldMax){
		return newDist;
	}
	return oldMax;
}

/*set new values via reduction on the thread level
 *
 * @param  oldVal current maximum radius from the selected point
 * @param  newVal calculated new radius for the current selected point
 *
 */
__device__ void setNewValues(VectorResult *oldVal, VectorResult *newVal)
{
	if(oldVal->r < newVal->r){
		oldVal->i = newVal->i;
		oldVal->r = newVal->r;
	}
}

//reducing the result on the block level
/*set new values via reduction on the thread level
 *
 * @param  currR current maximum radius from the selected point
 * @param  finalVal calculated new radius for the next selected point
 *
 */
__device__ void resultReduction(VectorResult *currR, VectorResult *finalVal)
{
	if(currR->r == -1){
		currR->r = finalVal->r;
		currR->i = finalVal->i;
	} else{
		if(currR->r > finalVal->r){
			currR->r = finalVal->r;
			currR->i = finalVal->i;
		}
	}
}

/**
 * Device kernel to calculate metric center on the given input.
 * <P>
 * Called with a one-dimensional grid of one-dimensional blocks, Each block updates one values
 * current chosen point. The points are balanced across all given blocks. Each
 * thread within a block computes the distance with respect to the set cordinate in the block
 *
 * @param  pList  Array of input point list.
 * @param  rList  Array of result list storing all metric centers of each blocks.
 * @param  N     Number of input cordinates
 *
 * @author  Utkarsh Bhatia
 * @version 06-Nov-2015
 */
extern "C" __global__ void metricCalculate
(VectorInput *pList, VectorResult *rList, int N)
{
	int totalBlocks = gridDim.x; //total number of blocks
	int idBlock = blockIdx.x; //id of the current block
	int idThread = threadIdx.x; //id of the current thread

	//current tempIndex being processed in the block
	double tempIndex = 0;
	//dist of the current two points
	double dist = 0;
	//max distance of the current point in the block
	double maxDist=0.0;

	//compute and calculate distance of the the point w.r.t to every other cordinate
	for(int i = idBlock; i < N; i = i + totalBlocks) {
		tempIndex = i;
		maxDist=0.0;
		for(int j = idThread; j < N; j = j + NT) {
			dist = tempDistance(&pList[i], &pList[j]);
			maxDist = getMaxDist(maxDist, dist);
		
		}

		resultArray[idThread] = (VectorResult){maxDist,tempIndex};


		// Compute largest distance via shared memory parallel reduction.
		__syncthreads();
		for (int k = NT/2; k > 0; k >>= 1)
		{
			if (idThread < k)
			{
				VectorResult *temp1 = &resultArray[idThread];
				VectorResult *temp2 = &resultArray[idThread+k];
				//calling method to reduce the thread level reduction
				setNewValues(temp1, temp2);
			}
			__syncthreads();
		}

		// Single threaded section.
		if (idThread == 0)
		{
			//calling method to reduce the thread level reduction
			resultReduction(&rList[idBlock], &resultArray[0]);

		}	
	}
}

