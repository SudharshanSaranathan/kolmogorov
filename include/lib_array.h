#ifndef _LIBARRAY_
#define _LIBARRAY_

#include <limits>
#include <vector>
#include <cstdlib>
#include <complex>

/* ----------------
 * Define typedefs.
 * ----------------
 */

using sizt = std::size_t;
using ulng = unsigned long;
using cmpx = std::complex<double>;

template <typename type>
using limits = std::numeric_limits<type>;

template <typename type>
using type_vector = std::vector<type>;
using sizt_vector = std::vector<sizt>;
using long_vector = std::vector<long>;

/* -----------------------------------
 * Calculate total number of elements.
 * -----------------------------------
 */

sizt sizeof_vector(      sizt_vector&);
sizt sizeof_vector(const sizt_vector&);

/* --------------------
 * Template class Array
 * --------------------
 */

template <class type>
class Array{

/* ----------------------
 * Private class members.
 * ----------------------
 */

private:

/*
 * Variable declaration.
 * --------------------------------
 * Name     Type        Description
 * --------------------------------
 * dims     sizt_vector Dimensions of the array.
 * nans     bool        Does the array have NaNs?
 * stat     bool        Has memory been allocated on the heap?
 * size     sizt        Total number of elements in the array.
 */

    sizt_vector dims;
    bool        nans = false;
    bool        stat = false;
    sizt        size = 1;

/*
 * Pointers declaration.
 * ------------------------------------
 * Name         Type        Description
 * ------------------------------------
 * data_ptr_1D  type*       Pointer to 1D array.    
 * data_ptr_2D  type**      Pointer to 2D array.        
 * data_ptr_3D  type***     Pointer to 3D array.
 * data_ptr_4D  type****    Pointer to 4D array.
 * 
 * -------------------
 * Additional Comments
 * -------------------
 * The data_ptr..., only provide python-like access 
 * to array elements. The actual data is stored in
 * root_ptr, see below.
 */

    type*    data_ptr_1D = nullptr;
    type**   data_ptr_2D = nullptr;
    type***  data_ptr_3D = nullptr;
    type**** data_ptr_4D = nullptr;

public:
/*
 * Pointers declaration.
 * ------------------------------------
 * Name         Type        Description
 * ------------------------------------
 * root_ptr     type*       Pointer for 1D array
 * 
 * -------------------
 * Additional Comments
 * -------------------
 * root_ptr stores the actual data, irrespective of dimensions,
 * as a 1D contiguous array. 
 */

    type*    root_ptr    = nullptr;

public:
   ~Array();
    Array();
    Array(const sizt_vector&);
    Array(const Array<type>&);

public:
    bool        get_stat();
    sizt        get_size();
    sizt        get_dims(sizt);
    type	    get_total();
    sizt_vector get_dims();

public:
    type* operator[](const sizt);
    type* operator[](const sizt, const sizt);
    type* operator[](const sizt, const sizt, const sizt);
    type* operator[](const sizt, const sizt, const sizt, const sizt);

    type& operator()(const sizt);
    type& operator()(const sizt, const sizt);
    type& operator()(const sizt, const sizt, const sizt);
    type& operator()(const sizt, const sizt, const sizt, const sizt);

    Array<type>& operator =(      Array<type> );
    Array<type>  operator +(const Array<type>&);
    Array<type>  operator -(const Array<type>&);
    Array<type>  operator *(const Array<type>&);
    Array<type>  operator /(const Array<type>&);
    void         operator+=(const Array<type>&);
    void         operator-=(const Array<type>&);
    void         operator*=(const Array<type>&);
    void         operator/=(const Array<type>&);

    Array<type>  operator +(type);
    Array<type>  operator -(type);
    Array<type>  operator *(type);
    Array<type>  operator /(type);
    void         operator+=(type);
    void         operator-=(type);
    void         operator*=(type);
    void         operator/=(type);

public:
    template <typename TYPE>
    friend void swap(Array<TYPE>&, Array<TYPE>&);

    int rd_bin(const char*);
    int wr_bin(const char*,  bool clobber=false);

    int rd_fits(const char*);
    int wr_fits(const char*, bool clobber=false);
};

#endif
