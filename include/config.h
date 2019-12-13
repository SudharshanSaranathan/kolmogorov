#ifndef _CONFIG_
#define _CONFIG_

#include <string>

using string = std::string;
using uint   = unsigned int;

typedef struct config {
public:
    static string read_fried_from;
    static string read_basis_from;
    static string read_weights_from;
    static string read_fftwisdom_from;
    static string read_aperture_function_from;

    static string write_log_to;
    static string write_phase_to;
    static string write_residual_to;

    static uint   sims_per_fried;
    static uint   sims_size_x;
    static uint   sims_size_y;

    static double phase_size;
    static double aperture_size;
    static double aperture_sampling_factor;

    static bool   output_save;
    static bool   output_clobber;

} config;

int config_parse(const char*);

#endif
