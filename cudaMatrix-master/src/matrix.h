#ifndef _matrix_hu_
#define _matrix_hu_

#include "errorMacros.h"

class Matrix
{
private:
	float *h;
	float *d;
	unsigned int w;
	bool touched;
	bool _isLU;

	//! Copy matrix from host to device
	inline void copyHtoD()
	{
		CHECK_SUCCESS(cudaMemcpy(d, h, w*w*sizeof(float), cudaMemcpyHostToDevice));
		touched = 0;
	}
	//! Copy matrix from host to device
	inline void copyDtoH()
	{
		CHECK_SUCCESS(cudaMemcpy(h, d, w*w*sizeof(float), cudaMemcpyDeviceToHost));
		touched = 0;
	}

public:
	static const unsigned int MAX_SANE_WIDTH = 6;

	//! Allocate a matrix on host and device
	Matrix(unsigned int width);
	//! Free a matrix from host and device
	~Matrix();

	//! Fill matrix with integer values from -128 to 127
	void fill();

	//! Print a matrix
	void print();

	//! Copy the matrix
	Matrix *copy();

	//! LU Decomposition of a matrix
	void decomposeLU();

	//! Rebuild the original matrix from an LU
	void multiplyLU();

	//! Rebuild the inverse matrix from an inverse LU
	void multiplyUL();

	//! Multiply two matrices and store them in this one
	void multiply(Matrix *a, Matrix *b);

	//! Are the matrices different
	bool isDifferent(Matrix *m);

	//! Invert
	void invertLU();

	//! All the get/setters!
	inline float *getH() { return h; };
	inline float *getD() { return d; };
	inline unsigned int getW() { return w; }
	inline bool hasBeenTouched(){ return touched; }
	inline void touch(){ touched = true; }
	inline bool isLU() { return _isLU; }
	inline void setLU(bool a) { _isLU = a; }
};



#endif //_matrix_hu_
