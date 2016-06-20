//equation3.cpp: a simple implementation 
//of the python 'equation3' from the jpsth_2.py script
//in order to speed up calculation of the raw jpsth.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stdint.h"
#include <iostream>
#include "inttypes.h"
#include "Python.h"
#include "arrayobject.h"

#define LONGLONG int64_t 

static PyObject* equation3(PyObject* self, PyObject* args)
{
	//pointers to python objects
	PyObject* spike1;
	PyObject* spike2;
	PyObject* arr1;
	PyObject* arr2;
	PyObject* result;

	//get the arrays from python
	if (!PyArg_ParseTuple(args, "OO", &spike1, &spike2))
		return NULL;

	//convert to arrays
	arr1 = PyArray_FROM_OTF(spike1, NPY_DOUBLE, NPY_IN_ARRAY); 
	if (arr1 == NULL) return NULL;
	arr2 = PyArray_FROM_OTF(spike2, NPY_DOUBLE, NPY_IN_ARRAY);
	if (arr2 == NULL) goto fail;

	//get dims of input arrays
	npy_intp* dims = new npy_intp[2];
	dims[0] = PyArray_DIM(arr1, 1);
	dims[1] = PyArray_DIM(arr2, 1);
	int num_trials = (int)PyArray_DIM(arr1,0);
	//create the output array
	result = PyArray_ZEROS(2, dims, NPY_FLOAT, 0);

	//do the real work of the function
	for (int i = 0; i < (int)dims[0]; i++)
	{
		for (int j = 0; j < (int)dims[1]; j++)
		{
			sum = 0.0;
			for (k = 0; k < num_trials; k++)
			{
				
				sum += 
			}
			void* r_address = PyArray_GETPTR2(result,i,j);

		}
				
	}

	Py_DECREF(arr1);
	Py_DECREF(arr2);
	free(dims);

	return

	fail:
		Py_XDECREF(arr1);
		Py_XDECREF(arr2);

}