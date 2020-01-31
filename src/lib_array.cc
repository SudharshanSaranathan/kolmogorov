#include "fitsio.h"
#include "lib_mem.h"
#include "lib_array.h"

#include <ctime>
#include <cmath>
#include <string>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <exception>
#include <algorithm>

sizt sizeof_vector(      sizt_vector &vector){
    sizt N = 1;
    for(sizt ind = 0; ind < vector.size(); ind++){
        N *= vector[ind];
    }
    return(N);
}

sizt sizeof_vector(const sizt_vector &vector){
    sizt N = 1;
    for(sizt ind = 0; ind < vector.size(); ind++){
        N *= vector[ind];
    }
    return(N);
}

template <class type>
Array<type>:: Array(){
}

template <class type>
Array<type>::~Array(){

    memory<type>::deallocate(data_ptr_1D);
    memory<type>::deallocate(data_ptr_2D);
    memory<type>::deallocate(data_ptr_3D);
    memory<type>::deallocate(data_ptr_4D);

    this->dims.clear();
    this->stat = false;
    this->size = 0;
}

template <class type>
Array<type>:: Array(const sizt_vector &dimensions){

    this->dims = dimensions;
    this->size = sizeof_vector(this->dims);

    switch(this->dims.size()){
        case 1: this->data_ptr_1D = memory<type>::allocate(this->dims[0]);
                this->stat        = this->data_ptr_1D == nullptr ? false : true;
                this->root_ptr    = this->data_ptr_1D;
                break;

        case 2: this->data_ptr_2D = memory<type>::allocate(this->dims[0], this->dims[1]);
                this->stat        = this->data_ptr_2D == nullptr ? false : true;
                this->root_ptr    = this->data_ptr_2D[0];
                break;

        case 3: this->data_ptr_3D = memory<type>::allocate(this->dims[0], this->dims[1], this->dims[2]);
                this->stat        = this->data_ptr_3D == nullptr ? false : true;
                this->root_ptr    = this->data_ptr_3D[0][0];
                break;

        case 4: this->data_ptr_4D = memory<type>::allocate(this->dims[0], this->dims[1], this->dims[2], this->dims[3]);
                this->stat        = this->data_ptr_4D == nullptr ? false : true;
                this->root_ptr    = this->data_ptr_4D[0][0][0];
                break;
    }
}

template <class type>
Array<type>:: Array(const Array<type> &src){

    this->nans = src.nans;
    this->dims = src.dims;
    this->size = sizeof_vector(this->dims);

    switch(this->dims.size()){
        case 1: this->data_ptr_1D = memory<type>::allocate(this->dims[0]);
                this->stat        = this->data_ptr_1D == nullptr ? false : true;
                this->root_ptr    = this->data_ptr_1D;
                break;

        case 2: this->data_ptr_2D = memory<type>::allocate(this->dims[0], this->dims[1]);
                this->stat        = this->data_ptr_2D == nullptr ? false : true;
                this->root_ptr    = this->data_ptr_2D[0];
                break;

        case 3: this->data_ptr_3D = memory<type>::allocate(this->dims[0], this->dims[1], this->dims[2]);
                this->stat        = this->data_ptr_3D == nullptr ? false : true;
                this->root_ptr    = this->data_ptr_3D[0][0];
                break;

        case 4: this->data_ptr_4D = memory<type>::allocate(this->dims[0], this->dims[1], this->dims[2], this->dims[3]);
                this->stat        = this->data_ptr_4D == nullptr ? false : true;
                this->root_ptr    = this->data_ptr_4D[0][0][0];
                break;
    }

    if(this->stat == true && src.stat == true){
        std::memcpy(this->root_ptr, src.root_ptr, this->size * sizeof(type));
    }
}

template <class type>
bool         Array<type>:: get_stat(){
    return(this->stat);
}

template <class type>
sizt         Array<type>:: get_size(){
    return(this->size);
}

template <class type>
sizt         Array<type>:: get_dims(sizt xs){
    if(xs >= dims.size())
        return 0;
    else
        return(dims[xs]);   
}

template <class type>
type	     Array<type>:: get_total(){

    type total = static_cast<type>(0);
    for(sizt ind = 0; ind < this->size; ind++){
	    total += this->root_ptr[ind];
    }
    return(total);
}

template <class type>
sizt_vector  Array<type>:: get_dims(){
    return(this->dims);
}

template <class type>
type* Array<type>::operator[](const sizt xs){

    if(this->stat == false)
        return(nullptr);

    if(xs >= this->dims[0])
       return(nullptr);
    
    switch(this->dims.size()){

        case 1:  return(this->data_ptr_1D + xs);
        case 2:  return(this->data_ptr_2D[xs]);
        case 3:  return(this->data_ptr_3D[xs][0]);
        case 4:  return(this->data_ptr_4D[xs][0][0]);
        default: return(nullptr);

    }
}

template <class type>
type& Array<type>::operator()(const sizt xs){
    if(this->dims.size() != 1)
        throw std::runtime_error("In Array<type>::operator(), expected " + std::to_string(this->dims.size()) + " argument(s)");
    if(xs >= this->dims[0])
        throw std::range_error("array out of bounds");

    return(this->data_ptr_1D[xs]);
}

template <class type>
type& Array<type>::operator()(const sizt xs, const sizt ys){
    if(this->dims.size() != 2)
        throw std::runtime_error("In Array<type>::operator(), expected " + std::to_string(this->dims.size()) + " argument(s)");

    if(xs >= this->dims[0] || ys >= this->dims[1])
        throw std::range_error("array out of bounds");

    return(this->data_ptr_2D[xs][ys]);
}

template <class type>
type& Array<type>::operator()(const sizt xs, const sizt ys, const sizt zs){
    if(this->dims.size() != 3)
        throw std::runtime_error("In Array<type>::operator(), expected " + std::to_string(this->dims.size()) + " argument(s)");

    if(xs >= this->dims[0] || ys >= this->dims[1] || zs >= this->dims[2])
        throw std::range_error("array out of bounds");

    return(this->data_ptr_3D[xs][ys][zs]);
}

template <class type>
type& Array<type>::operator()(const sizt xs, const sizt ys, const sizt zs, const sizt ws){
    if(this->dims.size() != 4)
        throw std::runtime_error("In Array<type>::operator(), expected " + std::to_string(this->dims.size()) + " argument(s)");

    if(xs >= this->dims[0] || ys >= this->dims[1] || zs >= this->dims[2] || ws >this->dims[3])
        throw std::range_error("array out of bounds");

    return(this->data_ptr_4D[xs][ys][zs][ws]);
}

template <typename type>
void swap(Array<type>& src_a, Array<type>& src_b){

    std::swap(src_a.stat, src_b.stat);
    std::swap(src_a.size, src_b.size);
    std::swap(src_a.dims, src_b.dims);

    std::swap(src_a.root_ptr   , src_b.root_ptr);
    std::swap(src_a.data_ptr_1D, src_b.data_ptr_1D);
    std::swap(src_a.data_ptr_2D, src_b.data_ptr_2D);
    std::swap(src_a.data_ptr_3D, src_b.data_ptr_3D);
    std::swap(src_a.data_ptr_4D, src_b.data_ptr_4D);
}

template <class type>
Array<type>& Array<type>::operator= (Array<type> src){

    swap(*this, src);
    return(*this);
}

template <class type>
Array<type>  Array<type>::operator+ (const Array<type> &src){

    if(!this->stat || !src.stat)
        throw std::logic_error("expected allocated arguments");

    if(this->dims != src.dims)
        throw std::logic_error("expected matching dimensions");

    Array<type> sum(src.dims);
    for(sizt ind = 0; ind < src.size; ind++){
        sum.root_ptr[ind] = this->root_ptr[ind] + src.root_ptr[ind];
    }

    return(sum);
}

template <class type>
Array<type>  Array<type>::operator- (const Array<type> &src){

    if(!this->stat || !src.stat)
        throw std::logic_error("expected allocated arguments");

    if(this->dims != src.dims)
        throw std::logic_error("expected matching dimensions");

    Array<type> diff(src.dims);
    for(sizt ind = 0; ind < src.size; ind++){
        diff.root_ptr[ind] = this->root_ptr[ind] - src.root_ptr[ind];
    }

    return(diff);
}

template <class type>
Array<type>  Array<type>::operator* (const Array<type> &src){

    if(!this->stat || !src.stat)
        throw std::logic_error("In Array<type>::operator*(), expected allocated arguments");

    if(this->dims != src.dims)
        throw std::logic_error("In Array<type>::operator*(), expected matching dimensions");

    Array<type> product(src.dims);
    for(sizt ind = 0; ind < src.size; ind++){
        product.root_ptr[ind] = this->root_ptr[ind] * src.root_ptr[ind];
    }

    return(product);
}

template <class type>
Array<type>  Array<type>::operator/ (const Array<type> &src){

    if(!this->stat || !src.stat)
        throw std::logic_error("expected allocated arguments");

    if(this->dims != src.dims)
        throw std::logic_error("expected matching dimensions");

    Array<type> div(src.dims);
    type null = static_cast<type>(0);

    for(sizt ind = 0; ind < src.size; ind++){
        div.root_ptr[ind] = src.root_ptr[ind] == null ? std::numeric_limits<type>::infinity() : this->root_ptr[ind] / src.root_ptr[ind];
    }
    return(div);
}

template <class type>
void         Array<type>::operator+=(const Array<type> &src){
    
    if(!this->stat || !src.stat)
        throw std::logic_error("expected allocated arguments");

    if(this->dims != src.dims)
        throw std::logic_error("expected matching dimensions");

    for(sizt ind = 0; ind < src.size; ind++){
        this->root_ptr[ind] += src.root_ptr[ind];
    }
}

template <class type>
void         Array<type>::operator-=(const Array<type> &src){
    
    if(this->dims != src.dims)
        throw std::logic_error("expected matching dimensions");

    for(sizt ind = 0; ind < src.size; ind++){
        this->root_ptr[ind] -= src.root_ptr[ind];
    }
}

template <class type>
void         Array<type>::operator*=(const Array<type> &src){
    
    if(!this->stat || !src.stat)
        throw std::logic_error("expected allocated arguments");

    if(this->dims != src.dims)
        throw std::logic_error("expected matching dimensions");

    for(sizt ind = 0; ind < src.size; ind++){
        this->root_ptr[ind] *= src.root_ptr[ind];
    }
}

template <class type>
void         Array<type>::operator/=(const Array<type> &src){
    
    if(!this->stat || !src.stat)
        throw std::logic_error("expected allocated arguments");

    if(this->dims != src.dims)
        throw std::logic_error("expected matching dimensions");

    for(sizt ind = 0; ind < src.size; ind++)
        this->root_ptr[ind] /= src.root_ptr[ind] == static_cast<type>(0) ? std::numeric_limits<type>::infinity() : src.root_ptr[ind];
    
}

template <class type>
Array<type>  Array<type>::operator +(type value){

    if(!this->stat)
        throw std::logic_error("expected allocated arguments");

    Array<type> sum(this->dims);
    for(sizt ind = 0; ind < this->size; ind++){
        sum.root_ptr[ind] = this->root_ptr[ind] + value;
    }
    return(sum);
}

template <class type>
Array<type>  Array<type>::operator -(type value){

    if(!this->stat)
        throw std::logic_error("expected allocated arguments");

    Array<type> diff(this->dims);
    for(sizt ind = 0; ind < this->size; ind++){
        diff.root_ptr[ind] = this->root_ptr[ind] - value;
    }
    return(diff);
}

template <class type>
Array<type>  Array<type>::operator *(type value){

    if(!this->stat)
        throw std::logic_error("expected allocated arguments");

    Array<type> product(this->dims);
    for(sizt ind = 0; ind < this->size; ind++){
        product.root_ptr[ind] = this->root_ptr[ind] * value;
    }
    return(product);
}

template <class type>
Array<type>  Array<type>::operator /(type value){

    if(!this->stat)
        throw std::logic_error("expected allocated arguments");

    Array<type> div(this->dims);
    type null = static_cast<type>(0);

    if(value == null){
        for(sizt ind = 0; ind < this->size; ind++)
            div.root_ptr[ind] = std::numeric_limits<type>::infinity();

    }else{
        for(sizt ind = 0; ind < this->size; ind++)
            div.root_ptr[ind] = this->root_ptr[ind] / value;

    }
    
    return(div);
}

template <class type>
void         Array<type>::operator+=(type value){

    if(!this->stat)
        throw std::logic_error("In function Array<type>::operator+=(), cannot perform operation on empty array");

    for(sizt ind = 0; ind < this->size; ind++){
        this->root_ptr[ind] += value;
    }
}

template <class type>
void         Array<type>::operator-=(type value){

    if(!this->stat)
        throw std::logic_error("In function Array<type>::operator-=(), cannot perform operation on empty array");

    for(sizt ind = 0; ind < this->size; ind++){
        this->root_ptr[ind] -= value;
    }
}

template <class type>
void         Array<type>::operator*=(type value){

    if(!this->stat)
        throw std::logic_error("In function Array<type>::operator*=(), cannot perform operation on empty array");

    for(sizt ind = 0; ind < this->size; ind++){
        this->root_ptr[ind] *= value;
    }
}

template <class type>
void         Array<type>::operator/=(type value){

    if(!this->stat)
        throw std::logic_error("expected allocated arguments");

    type null = static_cast<type>(0);
    if(value == null){
        for(sizt ind = 0; ind < this->size; ind++)
            this->root_ptr[ind] = std::numeric_limits<type>::infinity();
    
    }else{
        for(sizt ind  = 0; ind < this->size; ind++)
            this->root_ptr[ind] /= value;

    }
}

template <class type>
Array<type>  Array<type>::slice(sizt index){

    if(this->stat == false)
        throw std::runtime_error("In function Array<type>::slice(), cannot slice empty array");

    if(index >= dims[0])
        throw std::range_error("In function Array<type>::slice(), array out of bounds");

    if(this->dims.size() == 1)
        throw std::range_error("In function Array<type>::slice(), cannot slice 1D array");

    sizt_vector dims_slice (this->dims.begin() + 1, this->dims.end());
    Array<type> array_slice(dims_slice);
    
    switch(this->dims.size()){
        
        case 2: memcpy(array_slice.root_ptr, this->data_ptr_2D[index], sizeof(type) * array_slice.get_size());
                break;
        
        case 3: memcpy(array_slice.root_ptr, this->data_ptr_3D[index], sizeof(type) * array_slice.get_size());
                break;
        
        case 4: memcpy(array_slice.root_ptr, this->data_ptr_4D[index], sizeof(type) * array_slice.get_size());
                break;
    }

    return(array_slice);
}

template <class type>
Array<type>  Array<type>::abs(){

    if(this->stat == false)
        throw std::runtime_error("In function Array<type>::abs(), cannot find absolute value of empty array");

    Array<type> absolute(this->dims);
    for(sizt ind = 0; ind < this->size; ind++)
        absolute.root_ptr[ind] = std::abs(this->root_ptr[ind]);

    return(absolute);

}

template <class type>
Array<type>  Array<type>::pad(sizt_vector dims_padded, sizt_vector dims_start, type pad_value){

    if(this->stat == false)
        throw std::runtime_error("In function Array<type>::pad(), cannot pad empty array");

    if(this->dims.size() != dims_start.size() || this->dims.size() != dims_padded.size())
        throw std::runtime_error("In function Array<type>::pad(), expected " + std::to_string(this->dims.size()) + "D vector(s) as argument(s)");

    for(sizt ind = 0; ind < this->dims.size(); ind++){
        if(this->dims[ind] + dims_start[ind] > dims_padded[ind])
            throw std::runtime_error("In function Array<type>::pad(), dimensions of the padded array are too small");
    }

    Array<type> array_padded(dims_padded);
    switch(this->dims.size()){

        case 1: for(sizt xpix = dims_start[0]; xpix < this->dims[0] + dims_start[0]; xpix++){
                    array_padded.data_ptr_1D[xpix] = this->data_ptr_1D[xpix - dims_start[0]];
                }
                break;

        case 2: for(sizt xpix = dims_start[0]; xpix < this->dims[0] + dims_start[0]; xpix++){
                    for(sizt ypix = dims_start[1]; ypix < this->dims[1] + dims_start[1]; ypix++){
                        array_padded(xpix, ypix) = this->data_ptr_2D[xpix - dims_start[0]][ypix - dims_start[1]];
                    }
                }
                break;

        case 3: for(sizt xpix = dims_start[0]; xpix < this->dims[0] + dims_start[0]; xpix++){
                    for(sizt ypix = dims_start[1]; ypix < this->dims[1] + dims_start[1]; ypix++){
                        for(sizt zpix = dims_start[2]; zpix < this->dims[2] + dims_start[2]; zpix++){
                            array_padded(xpix, ypix, zpix) = this->data_ptr_3D[xpix - dims_start[0]][ypix - dims_start[1]][zpix - dims_start[2]];                        
                        }
                    }
                }
                break;

        case 4: for(sizt xpix = dims_start[0]; xpix < this->dims[0] + dims_start[0]; xpix++){
                    for(sizt ypix = dims_start[1]; ypix < this->dims[1] + dims_start[1]; ypix++){
                        for(sizt zpix = dims_start[2]; zpix < this->dims[2] + dims_start[2]; zpix++){
                            for(sizt wpix = dims_start[3]; wpix < this->dims[3] + dims_start[3]; wpix++){
                                array_padded(xpix, ypix, zpix, wpix) = this->data_ptr_4D[xpix - dims_start[0]][ypix - dims_start[1]][zpix - dims_start[2]][wpix - dims_start[3]];
                            }
                        }
                    }
                }
                break;
    }
    
    return(array_padded);

}

template <class type>
Array<type>  Array<type>::roll(sizt_vector shift, bool clockwise){

    if(this->stat == false)
        throw std::runtime_error("In function Array<type>::roll(), cannot find absolute value of empty array");

    if(this->dims.size() != shift.size())
        throw std::runtime_error("In function Array<type>::roll(), empty argument");

    Array<type> array_rolled(this->dims);

    if(!clockwise){
        for(sizt ind = 0; ind < this->dims.size(); ind++){
            shift[ind] += (this->dims[ind] % 2);
        }
    }

    switch(shift.size()){

        case 1: for(sizt xpix = 0; xpix < this->dims[0]; xpix++){
                    array_rolled.data_ptr_1D[xpix] = this->data_ptr_1D[(xpix + shift[0]) % this->dims[0]];
                }
                break;

        case 2: for(sizt xpix = 0; xpix < this->dims[0]; xpix++){
                    for(sizt ypix = 0; ypix < this->dims[1]; ypix++){
                        array_rolled.data_ptr_2D[xpix][ypix] = this->data_ptr_2D[(xpix + shift[0]) % this->dims[0]]\
                                                                                [(ypix + shift[1]) % this->dims[1]];
                    }
                }
                break;
        
        case 3: for(sizt xpix = 0; xpix < this->dims[0]; xpix++){
                    for(sizt ypix = 0; ypix < this->dims[1]; ypix++){
                        for(sizt zpix = 0; zpix < this->dims[2]; zpix++){
                            array_rolled.data_ptr_3D[xpix][ypix][zpix] = this->data_ptr_3D[(xpix + shift[0]) % this->dims[0]]\
                                                                                          [(ypix + shift[1]) % this->dims[1]]\
                                                                                          [(zpix + shift[2]) % this->dims[2]];
                        }
                    }
                }
                break;
        
        case 4: for(sizt xpix = 0; xpix < this->dims[0]; xpix++){
                    for(sizt ypix = 0; ypix < this->dims[1]; ypix++){
                        for(sizt zpix = 0; zpix < this->dims[2]; zpix++){
                            for(sizt wpix = 0; wpix < this->dims[3]; wpix++){
                                array_rolled.data_ptr_4D[xpix][ypix][zpix][wpix] = this->data_ptr_4D[(xpix + shift[0]) % this->dims[0]]\
                                                                                                    [(ypix + shift[1]) % this->dims[1]]\
                                                                                                    [(zpix + shift[2]) % this->dims[2]]\
                                                                                                    [(wpix + shift[3]) % this->dims[3]];
                            }
                        }
                    }
                }
                break;
    }

    return(array_rolled);
}

template <typename type>
Array<type>  Array<type>::crop(sizt_vector dims_start, sizt_vector dims_type, bool vector_type){

    sizt_vector dims_end(this->dims.size());
    sizt_vector dims_sub(this->dims.size());

    if(this->dims.size() != dims_start.size() || this->dims.size() != dims_type.size())
        throw std::runtime_error("In function Array<type>::crop(), expected " + std::to_string(this->dims.size()) + "D vector(s)");

    if(vector_type){
        
        for(sizt ind = 0; ind < this->dims.size(); ind++){
            
            dims_sub[ind] = dims_type[ind];
            dims_end[ind] = dims_type[ind] + dims_start[ind];

            if(dims_end[ind] >= this->dims[ind])
                throw std::range_error("Array out of bounds\n");

        }

    }else{

        for(sizt ind = 0; ind < this->dims.size(); ind++){

            dims_end[ind] = dims_type[ind];
            if(dims_start[ind] >= dims_end[ind])
                throw std::runtime_error("Expected dims_start[" + std::to_string(ind) + "] <= dims_end[" + std::to_string(ind) + "]");
            else
                dims_sub[ind] = dims_end[ind] - dims_start[ind];

        }
    }

    Array<type> array_cropped(dims_sub);
    switch(this->dims.size()){

        case 1: for(sizt xpix = 0; xpix < dims_sub[0]; xpix++){
                    array_cropped(xpix) = this->data_ptr_1D[xpix + dims_start[0]];
                }
                break;

        case 2: for(sizt xpix = 0; xpix < dims_sub[0]; xpix++){
                    for(sizt ypix = 0; ypix < dims_sub[1]; ypix++){
                        array_cropped(xpix, ypix) = this->data_ptr_2D[xpix + dims_start[0]][ypix + dims_start[1]];
                    }
                }
                break;

        case 3: for(sizt xpix = 0; xpix < dims_sub[0]; xpix++){
                    for(sizt ypix = 0; ypix < dims_sub[1]; ypix++){
                        for(sizt zpix = 0; zpix < dims_sub[2]; zpix++){
                            array_cropped(xpix, ypix, zpix) = this->data_ptr_3D[xpix + dims_start[0]][ypix + dims_start[1]][zpix + dims_start[2]];                        
                        }
                    }
                }
                break;

        case 4: for(sizt xpix = 0; xpix < dims_sub[0]; xpix++){
                    for(sizt ypix = 0; ypix < dims_sub[1]; ypix++){
                        for(sizt zpix = 0; zpix < dims_sub[2]; zpix++){
                            for(sizt wpix = 0; wpix < dims_sub[3]; wpix++){
                                array_cropped(xpix, ypix, zpix, wpix) = this->data_ptr_4D[xpix + dims_start[0]][ypix + dims_start[1]][zpix + dims_start[2]][wpix + dims_start[3]];
                            }
                        }
                    }
                }
                break;
    }

    return(array_cropped);
}

template <class type>
int          Array<type>::rd_bin(const char *filename){

    std::ifstream file(filename, std::ios::binary | std::ios::in | std::ios::ate);
    sizt filesize = file.tellg();
    file.seekg(0, std::ios::beg);

    if(this->stat == false || this->size != (filesize / sizeof(type))){
        file.close();
        return(EXIT_FAILURE);
    }

    file.read(reinterpret_cast<char*>(this->root_ptr), filesize);
    file.close();

    return(EXIT_SUCCESS);
}

template <class type>
int          Array<type>::wr_bin(const char *filename, bool clobber){

    if(!clobber && !access(filename, F_OK))
        return(EXIT_FAILURE);

    std::ofstream file(filename, std::ios::binary | std::ios::out);
    file.write(reinterpret_cast<char*>(this->root_ptr), this->size * sizeof(type));
    file.close();

    return(EXIT_SUCCESS);
}

template <class type>
int          Array<type>::rd_fits(const char *filename){

    int this_bitpix   = 0;
    int this_datatype = 0;
    if(std::is_same<type, float>::value){

        this_bitpix   = -32;
        this_datatype = TFLOAT;

    }else if(std::is_same<type, double>::value){

        this_bitpix   = -64;
        this_datatype = TDOUBLE;

    }else if(std::is_same<type, std::complex<float>>::value){

        this_bitpix   = -32;
        this_datatype = TCOMPLEX;

    }else if(std::is_same<type, std::complex<double>>::value){

        this_bitpix   = -64;
        this_datatype = TDBLCOMPLEX;

    }else{

        return(EXIT_FAILURE);

    }

    fitsfile *file = nullptr;
    sizt count = 1;
    int n_axis = 0;
    int status = 0;
    int bitpix = 0;

    sizt_vector fpix;
    sizt_vector dims;

    fits_open_file(&file, filename, READONLY, &status);
    if(status != 0)
        return(status);
 
    fits_get_img_dim (file, &n_axis, &status); dims.resize(n_axis);
    fits_get_img_size(file,  n_axis, (long int*)dims.data(), &status);
    fits_get_img_type(file, &bitpix, &status);
    if(status != 0)
        return(status);

    std::reverse(dims.begin(), dims.end());
    Array<type> data(dims);
    if(data.get_stat() == false)
        return(EXIT_FAILURE);

    count = sizeof_vector(dims);
    fpix.resize(n_axis); std::fill(fpix.begin(), fpix.end(), 1);

    fits_read_pix(file, this_datatype, (long int*)fpix.data(), count, nullptr, data[0], nullptr, &status);
    if(status != 0)
        return(status);

    *this = data;
    
    fits_close_file(file, &status);
    return(EXIT_SUCCESS);
}

template <class type>
int 	     Array<type>::wr_fits(const char *name, bool clobber){

    int this_bitpix   = 0;
    int this_datatype = 0;

    if(std::is_same<type, float>::value){

        this_bitpix   = -32;
        this_datatype = TFLOAT;

    }else if(std::is_same<type, double>::value){

        this_bitpix   = -64;
        this_datatype = TDOUBLE;

    }else if(std::is_same<type, std::complex<float>>::value){

        this_bitpix   = -32;
        this_datatype = TCOMPLEX;

    }else if(std::is_same<type, std::complex<double>>::value){

        this_bitpix   = -64;
        this_datatype = TDBLCOMPLEX;

    }else{

        return(EXIT_FAILURE);

    }

/*
 * Variable declaration.
 * ----------------------------------------
 * Name	        Type            Description
 * ----------------------------------------
 * status	    int             Execution status of FITS routines. 
 * fileptr	    fitsfile        FITS file pointer.
 * filename	    std::string     FITS file name.
 * dimensions	std::vector     Dimensions of the array, reversed.
 */

    int         status  = 0;
    fitsfile   *fileptr = nullptr;
    std::string filename(name);
    sizt_vector dimensions = this->dims;

    filename = clobber == true ? "!" + filename : filename;
    std::reverse(std::begin(dimensions), std::end(dimensions));

/* -------------------------
 * Create FITS file pointer.
 * -------------------------
 */

    fits_create_file(&fileptr, filename.c_str(), &status);
    if(status != 0)
        return(status);

/* ------------------
 * Create FITS image.
 * ------------------
 */

    fits_create_img(fileptr, this_bitpix, dimensions.size(), (long int*)dimensions.data(), &status);
    if(status != 0)
        return(status);
    
/* -------------------
 * Write data to file.
 * -------------------
 */

    fits_write_img(fileptr, this_datatype, 1, this->size, this->root_ptr, &status);
    if(status != 0)
        return(status);
    
    fits_close_file(fileptr, &status);
    return(status);
}

template class Array<int>;
template class Array<cmpx>;
template class Array<float>;
template class Array<double>;
