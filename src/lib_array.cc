#include "fitsio.h"
#include "lib_mem.h"
#include "lib_array.h"

#include <cmath>
#include <string>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <stdexcept>
#include <exception>
#include <algorithm>

sizt sizeof_vector(sizt_vector &vector){
    sizt N = 1;
    for(int i = 0; i < vector.size(); i++){
        N *= vector[i];
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
        std::memcpy(this->root_ptr, src.root_ptr, this->size*sizeof(type));
    }
}

template <class type>
bool        Array<type>:: get_stat(){
    return(this->stat);
}

template <class type>
sizt        Array<type>:: get_size(){
    return(this->size);
}

template <class type>
sizt        Array<type>:: get_dims(sizt xs){
    if(xs >= dims.size())
        return 0;
    else
        return(dims[xs]);
    
}

template <class type>
sizt_vector Array<type>:: get_dims(){
    return(this->dims);
}

template <class type>
type& Array<type>::operator()(const sizt xs){
    if(this->dims.size() != 1)
        throw std::runtime_error("expected " + std::to_string(this->dims.size()) + " argument(s)");
    if(xs >= this->dims[0])
        throw std::range_error("array out of bounds");

    return(this->data_ptr_1D[xs]);
}

template <class type>
type& Array<type>::operator()(const sizt xs, const sizt ys){
    if(this->dims.size() != 2)
        throw std::runtime_error("expected " + std::to_string(this->dims.size()) + " argument(s)");

    if(xs >= this->dims[0] || ys >= this->dims[1])
        throw std::range_error("array out of bounds");

    return(this->data_ptr_2D[xs][ys]);
}

template <class type>
type& Array<type>::operator()(const sizt xs, const sizt ys, const sizt zs){
    if(this->dims.size() != 3)
        throw std::runtime_error("expected " + std::to_string(this->dims.size()) + " argument(s)");

    if(xs >= this->dims[0] || ys >= this->dims[1] || zs >= this->dims[2])
        throw std::range_error("array out of bounds");

    return(this->data_ptr_3D[xs][ys][zs]);
}

template <class type>
type& Array<type>::operator()(const sizt xs, const sizt ys, const sizt zs, const sizt ws){
    if(this->dims.size() != 4)
        throw std::runtime_error("expected " + std::to_string(this->dims.size()) + " argument(s)");

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
        throw std::logic_error("expected allocated arguments");

    if(this->dims != src.dims)
        throw std::logic_error("expected matching dimensions");

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
        div.root_ptr[ind] = src.root_ptr[ind] == null ? NAN : this->root_ptr[ind] / src.root_ptr[ind];
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

    for(sizt ind = 0; ind < src.size; ind++){
        this->root_ptr[ind] /= src.root_ptr[ind];
    }
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

    for(sizt ind = 0; ind < this->size; ind++){
        if(value == null)
            div.root_ptr[ind] = NAN;
        else
            div.root_ptr[ind] = this->root_ptr[ind] / value;
    }
    return(div);
}

template <class type>
void         Array<type>::operator+=(type value){

    if(!this->stat)
        throw std::logic_error("expected allocated arguments");

    for(sizt ind = 0; ind < this->size; ind++){
        this->root_ptr[ind] += value;
    }
}

template <class type>
void         Array<type>::operator-=(type value){

    if(!this->stat)
        throw std::logic_error("expected allocated arguments");

    for(sizt ind = 0; ind < this->size; ind++){
        this->root_ptr[ind] -= value;
    }
}

template <class type>
void         Array<type>::operator*=(type value){

    if(!this->stat)
        throw std::logic_error("expected allocated arguments");

    for(sizt ind = 0; ind < this->size; ind++){
        this->root_ptr[ind] *= value;
    }
}

template <class type>
void         Array<type>::operator/=(type value){

    if(!this->stat)
        throw std::logic_error("expected allocated arguments");

    type null = static_cast<type>(0);
    for(sizt ind = 0; ind < this->size; ind++){
        if(value == null)
            this->root_ptr[ind] = NAN;
        else
            this->root_ptr[ind] /= value;
    }
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

    fitsfile *file = nullptr;
    int n_axis = 0;
    int status = 0;

    sizt        count = 1;
    sizt_vector fpix;
    sizt_vector dims;

    fits_open_file(&file, filename, READONLY, &status);
    if(status != 0)
        return(status);

    fits_get_img_dim (file, &n_axis, &status); dims.resize(n_axis);
    fits_get_img_size(file,  n_axis, (long int*)dims.data(), &status);
    if(status != 0)
        return(status);

    std::reverse(dims.begin(), dims.end());
    Array<type> data(dims);
    if(data.get_stat() == false)
        return(EXIT_FAILURE);

    count = sizeof_vector(dims);
    fpix.resize(n_axis); std::fill(fpix.begin(), fpix.end(), 1);
    fits_read_pix(file, TDOUBLE, (long int*)fpix.data(), count, nullptr, data.root_ptr, nullptr, &status);
    if(status != 0)
        return(status);

    fits_close_file(file, &status);
    *this = data;
    return(EXIT_SUCCESS);
}

template class Array<cmpx>;
template class Array<double>;