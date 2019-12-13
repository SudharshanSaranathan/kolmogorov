#ifndef _LIBMEM_
#define _LIBMEM_

#include <cstdlib>

template <typename type>
class memory{
public:
  
  static type*    allocate(std::size_t xs){
    type* data = new type[xs];
    return(data);
  }

  static type**   allocate(std::size_t xs, std::size_t ys){
    type** data = new type*[xs];
    if(data == nullptr)
      return nullptr;

    data[0] = new type[xs*ys]();
    if(data[0] == nullptr){
      delete[] data;
      return nullptr;
    }

    for(std::size_t i=1; i<xs; i++){
      data[i] = data[i-1] + ys;
    }
    return data;
  }

  static type***  allocate(std::size_t xs, std::size_t ys, std::size_t zs){
    type ***data = new type**[xs]();
    if(data == nullptr)
      return nullptr;

    data[0] = new type*[xs*ys]();
    if(data[0] == nullptr){
      delete[] data;
     return nullptr;
    }

    data[0][0] = new type[xs*ys*zs]();
    if(data[0][0] == nullptr){
      delete[] data[0];
      delete[] data;
      return nullptr;
    }

    for(std::size_t i=0; i<xs; i++){
      *(data+i) = *data + i*ys;
      for(std::size_t j=0; j<ys; j++)
        *(*(data+i)+j) = **data + i*ys*zs + j*zs;
    }
    return data;
  }

  static type**** allocate(std::size_t xs, std::size_t ys, std::size_t zs, std::size_t ws){
    type ****data = new type***[xs]();
    if(data==nullptr)
      return(nullptr);

    data[0] = new type**[xs*ys]();
    if(data[0]==nullptr){
      delete[] data;
      return nullptr;
    }

    data[0][0] = new type*[xs*ys*zs]();
    if(data[0][0]==nullptr){
      delete[] data[0];
      delete[] data;
      return(nullptr);
    }

    data[0][0][0] = new type[xs*ys*zs*ws]();
    if(data[0][0][0] == nullptr){
     delete[] data[0][0];
     delete[] data[0];
      delete[] data;
     return nullptr;
   }

   for(std::size_t i=0; i<xs; i++){
     *(data+i) = *data + i*ys;
     for(std::size_t j=0; j<ys; j++){
        *(*(data+i)+j) = **data + i*ys*zs + j*zs;
       for(std::size_t k=0; k<zs; k++){
         *(*(*(data+i)+j)+k) = ***data + i*ys*zs*ws + j*zs*ws + k*ws;
       }
      }
   }
   return data;
  }

  static void deallocate(type**** data){
    if(data != nullptr){
      if(data[0] != nullptr){
        if(data[0][0] != nullptr){
          if(data[0][0][0] != nullptr){
            delete[] data[0][0][0];
          }
          delete[] data[0][0];
        }
        delete[] data[0];
      }
      delete[] data;
    }
  }

  static void deallocate(type***  data){
    if(data != nullptr){
      if(data[0] != nullptr){
        if(data[0][0] != nullptr){
          delete[] data[0][0];
        }
        delete[] data[0];
      }
      delete[] data;
    }
  }

  static void deallocate(type**   data){
    if(data != nullptr){
      if(data[0] != nullptr){
        delete[] data[0];
      }
      delete[] data;
    }
  }

  static void deallocate(type*    data){
    if(data != nullptr){
      delete[] data;
    }
  }
};

#endif
