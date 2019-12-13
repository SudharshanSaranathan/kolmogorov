#include "mpi.h"
#include "fftw3.h"
#include "fitsio.h"
#include "lib_mpi.h"
#include "lib_array.h"
#include "lib_phase.h"

#include <ctime>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <iostream>

int main(int argc, char *argv[]){

    MPI_Status status;
    int process_rank = 0;
    int processes_total = 0;
    int processes_active = 0;
    int processes_deleted = 0;
    int mpi_recv_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processes_total);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    FILE *console   = process_rank == 0 ? stdout : fopen("/dev/null","wb");
    fprintf(console, "------------------------------------------------------\n");
    fprintf(console, "- Turbulence-degraded phasescreen simulation program -\n");
    fprintf(console, "------------------------------------------------------\n");

    if(argc < 2){
        fprintf(console, "1. Config file required. Aborting!\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    fprintf(console, "1. Reading %s:\t", argv[1]);
    if(config_parse(argv[1]) == EXIT_FAILURE){
        fprintf(console, "[Failed]\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }else
    {
        fprintf(console, "[Done]\n");
    }
    
    if(process_rank == 0){

        long   fried_in_queue   = 0;
        long   fried_completed  = 0;
        double progress_percent = 0.0;

        Array<double> fried;
        Array<double> aperture;

//      Read the fried parameters from file.
        fprintf(console, "2. Reading file %s:\t", config::read_fried_from.c_str());
        fried.rd_fits(config::read_fried_from.c_str());
        fprintf(console, "[Done]\n");

//      Read the aperture function from file.
        fprintf(console, "3. Reading file %s:\t", config::read_aperture_function_from.c_str());
        aperture.rd_fits(config::read_aperture_function_from.c_str());
        fprintf(console, "[Done]\n");

	//      Create vectors to store frequently used dimensions  
        sizt_vector dims_phase{fried.get_size(), config::sims_per_fried, config::sims_size_x, config::sims_size_y};
        sizt_vector dims_phase_per_fried{config::sims_per_fried, config::sims_size_x, config::sims_size_y};
        sizt_vector process_fried_map; process_fried_map.resize(fried.get_size() + 1);
        
        for(int id = 1; id < processes_total; id++){
            if(id > fried.get_size()){
                MPI_Send(fried.root_ptr, 1, MPI_DOUBLE, id, mpi_cmds::shutdown, MPI_COMM_WORLD);
                MPI_Send(aperture.root_ptr, aperture.get_size(), MPI_DOUBLE, id, mpi_cmds::shutdown, MPI_COMM_WORLD);
                processes_total--;          
            }else{
                MPI_Send(fried.root_ptr + fried_in_queue, 1, MPI_DOUBLE, id, mpi_cmds::stayalive, MPI_COMM_WORLD);
                MPI_Send(aperture.root_ptr, aperture.get_size(), MPI_DOUBLE, id, mpi_cmds::stayalive, MPI_COMM_WORLD);
                process_fried_map[id] = fried_in_queue;
                fried_in_queue++;
            }
        }
        
    }else if(process_rank){

	sizt_vector dims_phase{sizt(config::phase_size * config::sims_size_x / config::aperture_size),\
			       sizt(config::phase_size * config::sims_size_y / config::aperture_size)};
  
        sizt_vector dims_aperture{config::sims_size_x, config::sims_size_y};
	sizt_vector dims_phase_per_fried{config::sims_per_fried, config::sims_size_x, config::sims_size_y};

        Array<double> aperture(dims_aperture);
	Array<double> phase_per_fried(dims_phase_per_fried);
	Array<cmpx>   phase(dims_phase);
	Array<cmpx>   phase_fourier(dims_phase);

        fftw_import_wisdom_from_filename(config::read_fftwisdom_from.c_str());
        fftw_plan reverse = fftw_plan_dft_2d(dims_phase[0], dims_phase[1],\
                                             reinterpret_cast<fftw_complex*>(phase_fourier.root_ptr),\
                                             reinterpret_cast<fftw_complex*>(phase.root_ptr),\
                                             FFTW_BACKWARD, FFTW_MEASURE);
    
	double aperture_radius = config::sims_size_x / (2.0 * config::aperture_sampling_factor);
        double fried = 0.0;

        sizt phase_center_x = sizt(config::sims_size_x / 2.0);
        sizt phase_center_y = sizt(config::sims_size_y / 2.0);

        sizt sims_center_x = sizt(dims_phase[0] / 2.0);
        sizt sims_center_y = sizt(dims_phase[1] / 2.0);

        MPI_Recv(&fried, 1,  MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(aperture.root_ptr, aperture.get_size(), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	fprintf(stdout, " - process %d received aperture\n", process_rank);
        /*
	while(status.MPI_TAG != mpi_cmds::shutdown){
            for(int l = 0; l < config::sims_per_fried; l++){
                create_phase_screen_fourier(phase_fourier, fried, config::aperture_sampling_factor*config::phase_size);
            
                fftw_execute_dft(reverse, reinterpret_cast<fftw_complex*>(phase_fourier.root_ptr),\
                                          reinterpret_cast<fftw_complex*>(phase.root_ptr));
            }
        }
	*/
    }

    MPI_Finalize();
    return(EXIT_SUCCESS);
}
