#ifndef _LIBARRAY_
#define _LIBARRAY_

#define _USE_APERTURE_
#define _USE_SINGLE_PRECISION_

#include "fitsio.h"

#include <limits>
#include <vector>
#include <cstdlib>
#include <complex>
#include <typeinfo>
#include <stdexcept>

/* ----------------
 * Define typedefs.
 * ----------------
 */

typedef float precision;

using ulng = unsigned long;
using sizt = std::size_t;
using cmpx = std::complex<double>;

template <typename type>
using limits = std::numeric_limits<type>;

template <typename type>
using type_vector = std::vector<type>;
using sizt_vector = std::vector<sizt>;
using long_vector = std::vector<long>;

/*
 * Function declaration
 * ------------------------------------------------
 * Name                 Return type     Description 
 * ------------------------------------------------
 * sizeof_vector()      sizt            Returns the product of the sizt_vector elements,
 *                                      used for temporary arguments.
 *
 * sizeof_vector(const) sizt            Returns the product of the sizt_vector elements,
 *                                      when a reference exists.
 */

sizt sizeof_vector(      sizt_vector&);
sizt sizeof_vector(const sizt_vector&);

/* --------------------
 * Template class Array
 * --------------------
 */

template <class type>
class Array{
private:

/*
 * Variable declaration (PRIVATE)
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
 * Pointers declaration (PRIVATE)
 * ------------------------------------
 * Name         Type        Description
 * ------------------------------------
 * data_ptr_1D  type*       1D pointer.  
 * data_ptr_2D  type**      2D array of pointers.       
 * data_ptr_3D  type***     3D array of pointers.
 * data_ptr_4D  type****    4D array of pointers.
 * root_ptr     type*       1D pointer to a contiguous array. 
 *
 * -------------------
 * Additional Comments
 * -------------------
 * The pointers data_ptr_1D, data_ptr_2D, data_ptr_3D, data_ptr_4D, only provide
 * python-like access to the underlying array elements. Actual data is stored in
 * root_ptr.
 */

    type*    data_ptr_1D = nullptr;
    type**   data_ptr_2D = nullptr;
    type***  data_ptr_3D = nullptr;
    type**** data_ptr_4D = nullptr;
    type*    root_ptr    = nullptr;

public:

/*
 * Constructors / destructor declaration (PUBLIC)
 * ----------------------------------------------------
 * Name         Type                        Description
 * ----------------------------------------------------
 * Array()     ~Array()                     Class destructor.
 * Array()      Array()                     Class constructor, default.
 * Array()      Array(const sizt_vector&)   Class constructor, dimensions.
 * Array()      Array(const Array<type>&)   Class constructor, copy.
 */

   ~Array();
    Array();
    Array(const sizt_vector&);
    Array(const Array<type>&);

public:

/* 
 * Class methods declaration (PUBLIC)
 * See lib_array.cc for definitions
 *
 * ---------------------------------------------------
 * Name             Return type            Description
 * ---------------------------------------------------
 * get_stat()       bool            Returns true if instance is allocated / holds data.
 * get_sizt()       sizt            Returns total number of elements in the array.
 * get_total()      type            Returns the sum of all the elements.
 * get_dims(sizt)   sizt            Returns the dimension of the requested axis, throws exception if axis out of bounds.
 * get_dims()       sizt_vector     Returns a sizt_vector containing all dimensions.
 */

    bool        get_stat  ();
    sizt        get_size  ();
    type	    get_total ();
    sizt        get_dims  (sizt);
    sizt_vector get_dims  ();

public:

/*
 * Overloaded operators declaration (PUBLIC) 
 * See lib_array.cc for definitions
 * --------------------------------------------
 * Name             Return type     Description
 * --------------------------------------------
 * operator[]       type*           Returns a pointer to the requested index in the first axis.
 *                                  Mainly used in memcpy, for reading / writing slices of the 
 *                                  data at the specified index, throws exception if index out
 *                                  of bounds.
 * 
 * operator()       type&           Provides read / write access to individual array elements, 
 *                                  overloaded for 1D, 2D, 3D, and 4D arrays, throws exception 
 *                                  if indices out of bounds.
 *
 * operator=()      Array<type>&    Assignment operator that copies the argument into the array,
 *                                  using the copy-swap idiom. Dimensions of the argument array
 *                                  must match, throws exception if they don't.
 *
 * operator+()      Array<type>     Returns a new array that is the sum of the current array,
 *                                  and the argument. If argument is an array, dimensions must
 *                                  match, throws exception if they don't. If argument is a
 *                                  single value, adds the value to all elements.
 * 
 * operator-()      Array<type>     Returns a new array that is the subtraction of the argument
 *                                  from the current array. If argument is an array, dimensions
 *                                  must match, throws exception if they don't. If argument is
 *                                  a single value, subtracts the value from all elements.
 *                                  
 * operator*()      Array<type>     Returns a new array that is the product of the current array,
 *                                  and the argument. If argument is an array, dimensions must 
 *                                  match, throws exception if they don't. If argument is a value,
 *                                  multiplies the value with all elements.
 *
 * operator/()      Array<type>     Returns a new array that is the division of the current array,
 *                                  by the argument. If argument is an array, dimensions must
 *                                  match, throws exception if they don't. If argument is a single
 *                                  value, divides all elements with the value.
 *
 *                                  Note: If zeroes are encountered during division, the corresponding
 *                                        array elements in the return array are set to NANs.
 * 
 * operator+=()     void            Addition operator that adds the argument to the current array.
 *                                  If argument is an array, dimensions must match, throws exception
 *                                  if they don't. If argument is a single value, adds the value to
 *                                  all elements.
 *                       
 * operator-=()     void            Subtraction operator that subtracts the argument from the current
 *                                  array. If argument is an array, dimensions must match, throws 
 *                                  exception if they don't. If argument is a single value, subtracts 
 *                                  the value from all elements.
 *
 * operator*=()     void            Multiplication operator that multiplies the argument to the current 
 *                                  array. If argument is an array, dimensions must match, throws 
 *                                  exception if they don't. If argument is a single value, multiplies 
 *                                  the value to all elements.
 * 
 * operator/=()     void            Division operator that divides the current array by  the argument.
 *                                  If argument is an array, dimensions must match, throws exception
 *                                  if they don't. If argument is a single value, divides all elements
 *                                  by the value. 
 *
 *                                  Note: If zeroes are encountered during division, the corresponding
 *                                        array elements in the return array are set to NANs.
 *
 * -----------------------------------
 * Usage tip for overloaded operators:
 * -----------------------------------
 *
 * Use the shorthand operators (+=, -=, *=, /=) if a new array is not required i.e. instead of:
 *  
 * array_1 = array_1 + array_2, (or)
 * array_1 = array_1 * 3
 *
 * use:
 * 
 * array_1 += array_2, (or)
 * array_1 *= 3
 *
 * This GINORMOUSLY saves memory usage (and time).
 */
  
    type* operator[](const sizt);

    type& operator()(const sizt);
    type& operator()(const sizt, const sizt);
    type& operator()(const sizt, const sizt, const sizt);
    type& operator()(const sizt, const sizt, const sizt, const sizt);
    
    Array<type>& operator = (      Array<type> );
    Array<type>  operator + (const Array<type>&);
    Array<type>  operator - (const Array<type>&);
    Array<type>  operator * (const Array<type>&);
    Array<type>  operator / (const Array<type>&);
    void         operator+= (const Array<type>&);
    void         operator-= (const Array<type>&);
    void         operator*= (const Array<type>&);
    void         operator/= (const Array<type>&);

    Array<type>  operator + (type);
    Array<type>  operator - (type);
    Array<type>  operator * (type);
    Array<type>  operator / (type);
    void         operator+= (type);
    void         operator-= (type);
    void         operator*= (type);
    void         operator/= (type);
 
public:

/* 
 * Class methods declaration (PUBLIC)
 * See lib_array.cc for definitions
 * ----------------------------------------
 * Name         Type            Description
 * ----------------------------------------
 * abs  ()      Array<type>     Returns the std::abs() of the array.
 * pad  ()      Array<type>     Returns the array padded with zeros.
 * roll ()      Array<type>     Returns a cyclically rolled array.
 * crop ()      Array<type>     Returns a subset of the array.
 * slice()      Array<type>     Returns a slice of the array, at the specified index.
 */

    Array<type> slice (sizt);
    Array<type> abs   ();
    Array<type> pad   (sizt_vector, sizt_vector, type pad_value = static_cast<type>(0));
    Array<type> roll  (sizt_vector, bool clockwise = true);
    Array<type> crop  (sizt_vector, sizt_vector, bool vector_type = true);

public:

/*
 * Class methods declaration (PUBLIC)
 * See lib_array.cc for definitions
 * --------------------------------------------
 * Name             Return type     Description
 * --------------------------------------------
 * swap()           void            Friend function to swap the data of two instances.
 *                                  Required for the copy-swap idiom.
 *
 * rd_bin()         int             Read into array elements from a binary file. This
 *                                  is the fastest option.
 *
 * wr_bin()         int             Write the array elements to a binary file. This is
 *                                  the fastest option.*
 *
 * cast_to_type()   void            Life saving type-casting function for the class. Casts the elements
 *                                  of the array into the type of the argument, stores the type-casted
 *                                  values in the argument. Dimensions must match, throws exception if 
 *                                  they don't. See inline for definition.
 * 
 * cast_to_type()   Array<TYPE>     Life saving type-casting function for the class. Casts the elements
 *                                  of the array into the type of the argument, stores the type-casted
 *                                  values in the argument. Dimensions must match, throws exception if 
 *                                  they don't. See inline for definition.
 */

    template <typename TYPE>
    friend void swap(Array<TYPE>&, Array<TYPE>&);

    int rd_bin(const char*);
    int wr_bin(const char*,  bool clobber=false);

    template <typename TYPE>
    void cast_to_type(Array<TYPE>& array_to_cast_to){

    /* ---------------------------------------------
     * Check if dimensions of the input array match.
     * ---------------------------------------------
     */

        sizt_vector store_dims = array_to_cast_to.get_dims();
        if(this->dims.size() != store_dims.size())
            throw std::runtime_error("In function Array<type>::cast_to_type(), expected " + std::to_string(this->dims.size()) + "D argument");

        for(sizt ind = 0; ind < this->dims.size(); ind++){
            if(this->dims[ind] != store_dims[ind])
                throw std::runtime_error("In function Array<type>::cast_to_type(), expected dims[" + std::to_string(ind) + "] = " + std::to_string(this->dims[ind]) + " of argument");
        }

        if(this->stat == false)
            throw std::runtime_error("In function Array<type>::cast_to_type(), cannot cast an empty array");

    /* -----------------------
     * Cast the array to TYPE.
     * -----------------------
     * If the array being casted is of type std::complex<>, pick only the real part.
     */ 

        for(sizt ind = 0; ind < this->size; ind++)
            *(array_to_cast_to[0] + ind) = static_cast<TYPE>(std::real(this->root_ptr[ind]));

    }

    template <typename TYPE>
    Array<TYPE> cast_to_type(){        

        Array<TYPE> array_to_cast_to(this->dims);
        if(this->stat == false)
            return array_to_cast_to;

    /* -----------------------
     * Cast the array to TYPE.
     * -----------------------
     * If the array being casted is of type std::complex<>, pick only the real part.
     */

        for(sizt ind = 0; ind < this->size; ind++)
            *(array_to_cast_to[0] + ind) = static_cast<TYPE>(std::real(this->root_ptr[ind]));
        
        return(array_to_cast_to);
    }

/*
 * Class methods declaration (PUBLIC)
 * See lib_array.cc for definitions
 * ------------------------------------
 * Name             Type    Description
 * ------------------------------------
 * rd_fits()        int     Read into array elements from FITS file, requires
 *                          the "cfitsio" library. Can be called on allocated
 *                          or empty instance. If read failed, fitsio error
 *                          codes are returned:
 *
 *                          https://heasarc.gsfc.nasa.gov/docs/software/fitsio/quick/node26.html
 *
 *                          A few frequently encountered error codes are listed below:
 *                          ----------------------------------------------------------
 *                          Err code = 101 (Input and output are the same file)
 *                          Err code = 104 (If the file could not be read. For example, it doesn't exist)
 *                          Err code = 108 (Error reading from FITS file)
 *                          ----------------------------------------------------------
 *
 * wr_fits()        int     Writes array elements to FITS file, requires the 
 *                          "cfitsio" library. If write failed, fitsio error
 *                          codes are returned:
 *
 *                          https://heasarc.gsfc.nasa.gov/docs/software/fitsio/quick/node26.html
 *
 *                          A few frequently encountered error codes are listed below:
 *                          ----------------------------------------------------------
 *                          Err code = 101 (Input and output are the same file)
 *                          Err code = 105 (If th file could not be created. For example, the filename includes a directory path that does not exist)
 *                          Err code = 106 (Error writing to FITS file)
 *                          ----------------------------------------------------------
 */

    int rd_fits(const char*);
    int wr_fits(const char*, bool clobber=false);

};

#endif
