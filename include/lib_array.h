#ifndef _LIBARRAY_
#define _LIBARRAY_

#include <limits>
#include <vector>
#include <cstdlib>
#include <complex>

using sizt = std::size_t;
using ulng = unsigned long;
using cmpx = std::complex<double>;

template <typename type>
using limits = std::numeric_limits<type>;

template <typename type>
using type_vector = std::vector<type>;
using sizt_vector = std::vector<sizt>;
using long_vector = std::vector<long>;

sizt sizeof_vector(      sizt_vector&);
sizt sizeof_vector(const sizt_vector&);

template <class type>
class Array{
private:
    bool        nans = false;
    bool        stat = false;
    sizt        size = 1;
    sizt_vector dims;

    type*    data_ptr_1D = nullptr;
    type**   data_ptr_2D = nullptr;
    type***  data_ptr_3D = nullptr;
    type**** data_ptr_4D = nullptr;

public:
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
    type	get_total();
    sizt_vector get_dims();


public:
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
