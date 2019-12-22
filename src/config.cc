#include "config.h"
#include <fstream>
#include <sstream>

string config::read_fried_from              = "fried.fits";
string config::read_basis_from              = "basis.fits";
string config::read_weights_from            = "weights.fits";
string config::read_fft_psf_wisdom_from     = "fftw_wisdom_psf";
string config::read_fft_phase_wisdom_from   = "fftw_wisdom_phase";
string config::read_aperture_function_from  = "pupil.fits";

string config::write_log_to      = "log.file";
string config::write_phase_to    = "phase.fits";
string config::write_residual_to = "residual.fits";
string config::write_psf_to      = "psf.fits";

uint   config::sims_per_fried = 400;
uint   config::sims_size_x    = 94;
uint   config::sims_size_y    = 94;

double config::phase_size               = 10.0;
double config::aperture_size            = 1.0;
double config::aperture_sampling_factor = 1.0;

bool   config::output_save    = true;
bool   config::output_clobber = false;
bool   config::get_airy_disk  = false;

int config_parse(const char* filename){
  
    std::ifstream infile(filename);
    std::string line;
    if(!infile)
        return(EXIT_FAILURE);

    while(std::getline(infile, line)){
    
        std::stringstream tokens(line);
        std::string key, value;
        std::getline(tokens, key, ':');
        tokens >> std::ws;
        std::getline(tokens, value, ':');

        if(key == "fried")
	        config::read_fried_from = value;
        else if(key == "basis")
	        config::read_basis_from = value;
        else if(key == "aperture")
	        config::read_aperture_function_from = value;
        else if(key == "weights")
	        config::read_weights_from = value;
        else if(key == "fftw_psf")
	        config::read_fft_psf_wisdom_from = value;
        else if(key == "fftw_phase")
	        config::read_fft_phase_wisdom_from = value;
        else if(key == "log")
	        config::write_log_to = value;
        else if(key == "phase")
	        config::write_phase_to = value;
        else if(key == "residual")
	        config::write_residual_to = value;
        else if(key == "psf")
	        config::write_psf_to = value;
        else if(key == "realizations")
	        config::sims_per_fried = std::stoi(value);
        else if(key == "size_x")
	        config::sims_size_x = std::stoi(value);
        else if(key == "size_y")
	        config::sims_size_y = std::stoi(value);
        else if(key == "phase_size")
	        config::phase_size  = std::stof(value);
        else if(key == "aperture_size")
	        config::aperture_size = std::stof(value);
        else if(key == "aperture_sampling")
	        config::aperture_sampling_factor = std::stof(value);
        else if(key == "save")
	        config::output_save = value == "Y";
        else if(key == "clobber")
	        config::output_clobber = value == "Y"; 
        else if(key == "airy_disk")
	        config::get_airy_disk = value == "Y";
    }
    
    return(EXIT_SUCCESS);

}
