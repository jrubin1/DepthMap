/*
 * Project 2: Performance Optimization
 */

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "utils.h"
#include "calc_depth_naive.h"
#include "calc_depth_optimized.h"

#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

// copy the functions inline
/* Implements the displacement function */
float displacement_opt(int dx, int dy) {
    return sqrt(dx*dx+dy*dy);
}

/* Helper function to return the square euclidean distance between two values. */
float square_euclidean_distance_opt(float a, float b) {
    int diff = a - b;
    return diff * diff;
}

void calc_depth_optimized(float *depth, float *left, float *right,
        int image_width, int image_height, int feature_width,
        int feature_height, int maximum_displacement) {
    // Naive implementation

    #pragma omp parallel
    {
    #pragma omp for
    for (int y = 0; y < image_height; y++) {
        for (int x = 0; x < image_width; x++) {
            // checks for correct boundaries
            if (y < feature_height || y >= image_height - feature_height
                    || x < feature_width || x >= image_width - feature_width) {
                depth[y * image_width + x] = 0;
                continue;
            }
	    float min_diff = -1;
            int min_dy = 0;
            int min_dx = 0;
            for (int dy = -maximum_displacement; dy <= maximum_displacement; dy++) {
                for (int dx = -maximum_displacement; dx <= maximum_displacement; dx++) {
                    if (y + dy - feature_height < 0
                            || y + dy + feature_height >= image_height
                            || x + dx - feature_width < 0
                            || x + dx + feature_width >= image_width) {
                        continue;
                    }
                    float squared_diff = 0;
                    for (int box_y = -feature_height; box_y <= feature_height; box_y++) {
                         // Original for loop that was vectorized with SIMD instructions
			 /* for (int box_x = -feature_width; box_x <= feature_width; box_x++) {
                            int left_x = x + box_x;
                            int left_y = y + box_y;
                            int right_x = x + dx + box_x;
                            int right_y = y + dy + box_y;
                            squared_diff += square_euclidean_distance_opt(
                                    left[left_y * image_width + left_x],
                                    right[right_y * image_width + right_x]
                                    );
                        }*/
			
			// start vectorization
			float result[4];
			__m128 v_squared_diff = _mm_setzero_ps();
			int box_x = -feature_width;
			float *left_start = left+((y+box_y)*image_width)+x+box_x; // left pointer starting point
			float *right_start = right+((y+dy+box_y)*image_width)+x+dx+box_x; // right pointer starting point
 			

			// vectorization pulls in four floats at a time
            // #pragma omp parallel 
            // {
            //     #pragma omp for
        		for (int i = 0; i < (feature_width*2+1)/8*8; i += 8) {
        			// Original image access parameters
        			/*int left_x = x + (box_x + i);
        			int left_y = y + box_y;
        			int right_x = x + dx + (box_x + i);
        			int right_y = y + dy + box_y;
        			*/
        			
        			// vectorize square euclidean calculation
        			__m128 diff = _mm_sub_ps(_mm_loadu_ps(left_start+i), _mm_loadu_ps(right_start+i));
        			__m128 m_result = _mm_mul_ps(diff, diff);
        			v_squared_diff = _mm_add_ps(v_squared_diff, m_result);

                    diff = _mm_sub_ps(_mm_loadu_ps(left_start+i+4), _mm_loadu_ps(right_start+i+4));
                    m_result = _mm_mul_ps(diff, diff);
                    v_squared_diff = _mm_add_ps(v_squared_diff, m_result);
        		}
            // }

			// tail case
			_mm_storeu_ps(result, v_squared_diff);
			for (int i = (feature_width*2+1)/8*8; i < 2*feature_width+1; i++) {
				// do square euclidean calculation in line (slight program speedup)
                                float difference = *(left_start+i)-*(right_start+i);
				// append tail case results to result[0] because addition is commutative
				result[0] += difference*difference;
			}
			// add to square_diff
			squared_diff += result[0]+result[1]+result[2]+result[3];
		    // end vectorization
                    }
                    if (min_diff == -1 || min_diff > squared_diff
                            || (min_diff == squared_diff
                                && displacement_opt(dx, dy) < displacement_opt(min_dx, min_dy))) {
                        min_diff = squared_diff;
                        min_dx = dx;
                        min_dy = dy;
                    }
                }
            }
            if (min_diff != -1) {
                if (maximum_displacement == 0) {
                    depth[y * image_width + x] = 0;
                } else {
                    depth[y * image_width + x] = displacement_opt(min_dx, min_dy);
                }
            } else {
                depth[y * image_width + x] = 0;
            }
        }
    }
}
}
