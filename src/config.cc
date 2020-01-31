#include "config.h"
#include <fstream>
#include <sstream>

string io_t::read_image_from             = "image.fits";
string io_t::read_fried_from             = "fried.fits";
string io_t::read_basis_from             = "basis.fits";
string io_t::read_weights_from           = "weights.fits";
string io_t::read_fft_psf_wisdom_from    = "fftw_wisdom_psf";
string io_t::read_fft_phase_wisdom_from  = "fftw_wisdom_phase";
string io_t::read_aperture_function_from = "pupil.fits";

string io_t::write_log_to      = "log.file";
string io_t::write_phase_to    = "phase.fits";
string io_t::write_images_to   = "image_convolved.fits";
string io_t::write_residual_to = "residual.fits";
string io_t::write_psf_to      = "psf.fits";

bool   io_t::save    = true;
bool   io_t::clobber = false;

sizt      sims_t::realizations_per_fried = 400;
sizt      sims_t::size_x_in_pixels       = 92;
sizt      sims_t::size_y_in_pixels       = 92;
precision sims_t::size_in_meters         = 10.0;

precision aperture_t::size            = 1.0;
precision aperture_t::sampling_factor = 1.5;
bool      aperture_t::airy_disk       = false;

float image_t::original_sampling = 1.0;
float image_t::degraded_sampling = 1.0;

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

        if(key == "image")
	        io_t::read_image_from = value;

        else if(key == "fried")
	        io_t::read_fried_from = value;

        else if(key == "basis")
	        io_t::read_basis_from = value;

        else if(key == "aperture")
	        io_t::read_aperture_function_from = value;

        else if(key == "weights")
	        io_t::read_weights_from = value;
        
        else if(key == "fftw_psf")
	        io_t::read_fft_psf_wisdom_from = value;
        
        else if(key == "fftw_phase")
	        io_t::read_fft_phase_wisdom_from = value;
        
        else if(key == "log")
	        io_t::write_log_to = value;
        
        else if(key == "phase")
	        io_t::write_phase_to = value;
        
        else if(key == "convolved_images")
	        io_t::write_images_to = value;
        
        else if(key == "residual")
	        io_t::write_residual_to = value;
        
        else if(key == "psf")
	        io_t::write_psf_to = value;
        
        else if(key == "realizations")
	        sims_t::realizations_per_fried = std::stoi(value);
        
        else if(key == "phase_size_x_in_pixels")
	        sims_t::size_x_in_pixels = std::stoi(value);
        
        else if(key == "phase_size_y_in_pixels")
	        sims_t::size_y_in_pixels = std::stoi(value);
        
        else if(key == "phase_size_in_meters")
	        sims_t::size_in_meters  = std::stof(value);
        
        else if(key == "aperture_size_in_meters")
	        aperture_t::size = std::stof(value);
        
        else if(key == "aperture_sampling")
	        aperture_t::sampling_factor = std::stof(value);
        
        else if(key == "airy_disk")
	        aperture_t::airy_disk = value == "true";
        
        else if(key == "original_sampling")
	        image_t::original_sampling = std::stof(value);
    
        else if(key == "degraded_sampling")
	        image_t::degraded_sampling = std::stof(value);
        
        else if(key == "save")
	        io_t::save = value == "true";
        
        else if(key == "clobber")
	        io_t::clobber = value == "true"; 
    
    }
    
    return(EXIT_SUCCESS);

}
