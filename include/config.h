#ifndef _CONFIG_
#define _CONFIG_

#include <string>

using string = std::string;
using uint   = unsigned int;

typedef struct io_t{
public:
    
    static string read_image_from;
    static string read_fried_from;
    static string read_basis_from;
    static string read_weights_from;
    static string read_fft_psf_wisdom_from;
    static string read_fft_phase_wisdom_from;
    static string read_aperture_function_from;

    static string write_log_to;
    static string write_phase_to;
    static string write_images_to;
    static string write_residual_to;
    static string write_psf_to;
    
    static bool   save;
    static bool   clobber;

} io_t;

typedef struct sims_t{
public:
    
    static sizt  realizations_per_fried;
    static sizt  size_x_in_pixels;
    static sizt  size_y_in_pixels;
    static float sizt_in_meters;
} sims_t;

typedef struct aperture_t{
public:

    static float size;
    static float sampling_factor;
    static bool  airy_disk;

} aperture_t;

typedef struct image_t{
public:
    
    static float original_sampling;
    static float degraded_sampling;
} image_t;

int config_parse(const char*);

#endif
