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

/*
 * Description:
 *
 * This is a program to simulate atmospheric phase-screens, which are functions of \\
 * the fried parameter (Fried, 1967). A phase-screen is computed by taking the inverse \\
 * fourier transformation of an array (of spatial frequencies) that satisfies the \\
 * Kolmogorov-Obukhov power law.
 *
 * Program logic:
 * 
 * 1. Read and parse the config file. 
 * 2. Read fried parameters from FITS file.
 * 3. Send one fried parameter to each worker.
 * 4. Wait for workers to return simulation.
 * 5. Store the data, repeat steps 3-5.
 * 6. Write phase array to file.
 *
 * Additional information:
 *
 * Each comment block has a title (of sorts).
 * Titles starting with !, followed by a number 'n' indicate a block handling step 'n'.
 *
 */

int main(int argc, char *argv[]){

/*
 *  Declare MPI variables.
 *  
 *  Name		Type		Purpose
 *  status:		MPI_status	Required in MPI functions, see MPI documentation for explanation.
 *  process_rank:	int		Variable to store the rank of each MPI process.
 *  process_total:	int		Variable to store the number of processes requested with mpiexec.
 *  processes_active:	int		Variable to store the number of processes actually used by the program.
 *  processes_deleted:	int		Variable to store the number of processes that are not required by the program.
 *  mpi_recv_count:	int		Variable to store the count of data received in MPI_Recv, see MPI documentation for explanation. 
 *  
 */
   
    MPI_Status status;
    int process_rank = 0;
    int processes_total = 0;
    int processes_active = 0;
    int processes_deleted = 0;
    int mpi_recv_count = 0;


/*
 * Initialize MPI, get processes information.
 */
   
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processes_total);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

/*
 * Only the master MPI process - rank zero - prints to stdout. All others shut up.
 */

    FILE *console   = process_rank == 0 ? stdout : fopen("/dev/null","wb");
    fprintf(console, "------------------------------------------------------\n");
    fprintf(console, "- Turbulence-degraded phasescreen simulation program -\n");
    fprintf(console, "------------------------------------------------------\n");

/*
 * !(1) Read and parse the config file.
 */

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
	fprintf(console, " - wavefront size , in pixels:\t[%d %d]\n", config::sims_size_x, config::sims_size_y);
	fprintf(console, " - simulation size, in meters:\t[%lf]\n", config::phase_size);
        fprintf(console, " - aperture sampling factor:\t[%lf]\n", config::aperture_sampling_factor);
    }
    

/*
 * Workflow for the master MPI process.
 */

    if(process_rank == 0){
	
    /*
     * Initialize variables for distributing work.
     *
     * Name			Type	Purpose
     * index_of_fried_in_queue	long	Variable to store array index of the next fried parameter.
     * fried_completed		long	Variable to store the number of fried parameters simulated.
     * progress_percent		double	Variable to display the progress percentage.
     */

        long   index_of_fried_in_queue   = 0;
        long   fried_completed  = 0;
        double progress_percent = 0.0;

    /*
     * Initialize arrays.
     *
     * Name		Type		Purpose
     * fried		Array<double>	Array to store the fried parameters, see "lib_array.h" for datatype.
     * aperture		Array<double>	Array to store the aperture function, see "lib_array.h" for datatype.
     */

        Array<double> fried;
        Array<double> aperture;

    /*
     * !(2) Read fried parameters from file, store in array.
     */

	fprintf(console, "2. Reading file %s:\t", config::read_fried_from.c_str());
        fried.rd_fits(config::read_fried_from.c_str());
        fprintf(console, "[Done]\n");

    /*
     * Read aperture function from file, store in array.
     */
        
	fprintf(console, "3. Reading file %s:\t", config::read_aperture_function_from.c_str());
        aperture.rd_fits(config::read_aperture_function_from.c_str());
        fprintf(console, "[Done]\n");

    /*
     * Initialize variables to store dimensions, and maps.
     *
     * Name			Type			Purpose
     * dims_phase		std::vector<size_t>	Vector to store the dimensions of phase-screens, in pixels.
     * dims_phase_per_fried	std::vector<size_t>	Vector to store the dimensions of phase-screens, per fried, in pixels.
     * process_fried_map	std::vector<size_t>	Vector to store the map of which process is handling which fried parameter.
     */
    
    	sizt_vector dims_phase{fried.get_size(), config::sims_per_fried, config::sims_size_x, config::sims_size_y};
        sizt_vector dims_phase_per_fried{config::sims_per_fried, config::sims_size_x, config::sims_size_y};
        sizt_vector process_fried_map; process_fried_map.resize(fried.get_size() + 1);
       
    /*
     * Initialize MPI processes, shutdown if more MPI processes than fried parameters.
     *
     * Name	Type	Purpose
     * id	int	Variable to iterate over MPI processes.
     */

        for(int id = 1; id < processes_total; id++){
            if(id > fried.get_size()){

	    /*
	     * Kill MPI process if rank > fried.get_size(), data sent is irrelevant.
	     */

                MPI_Send(fried.root_ptr, 1, MPI_DOUBLE, id, mpi_cmds::shutdown, MPI_COMM_WORLD);
                MPI_Send(aperture.root_ptr, aperture.get_size(), MPI_DOUBLE, id, mpi_cmds::shutdown, MPI_COMM_WORLD);

	    /*
	     * Update number of processes in use.
	     */
		
		processes_total--;          

            }else{

	    /*
	     * Send fried parameter, and aperture function to process.
	     */

                MPI_Send(fried.root_ptr + index_of_fried_in_queue, 1, MPI_DOUBLE, id, mpi_cmds::stayalive, MPI_COMM_WORLD);
                MPI_Send(aperture.root_ptr, aperture.get_size(), MPI_DOUBLE, id, mpi_cmds::stayalive, MPI_COMM_WORLD);
                
	    /*
	     * Store index_of_fried_in_queue in process_fried_map[id].
	     * Process with rank id is now working on fried[index_of_fried_in_queue].
	     * Increment index_of_fried_in_queue.
	     */

		process_fried_map[id] = index_of_fried_in_queue;
                index_of_fried_in_queue++;
            }
        }

    /*
     * Initialize phase-screens.
     *
     * Name	Type		Purpose
     * phase	Array<double>	Array to store the simulated phase screens, later written to file.
     */

	Array<double> phase(dims_phase);

    /*
     * !(3, 4, 5) Distribute fried parameters to workers, store returned simulations.
     */

        while(fried_completed < fried.get_size()){
        	  
	/*
	 * Wait for a worker that is ready. If found, get and store worker info.
	 */	
	
	    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	    MPI_Get_count(&status, MPI_DOUBLE, &mpi_recv_count);

	/*
	 * Get the fried parameter index that the process was working on, from process_fried_map.
	 *
	 * Name		Type		Purpose
	 * fried_index	std::size_t	Variable to store the index of the fried parameter.
	 */

	    sizt fried_index = process_fried_map[status.MPI_SOURCE] * sizeof_vector(dims_phase_per_fried);

	/*
	 * Get data, and store in phase at the correct location.
	 */

	    MPI_Recv(phase.root_ptr + fried_index, mpi_recv_count, MPI_DOUBLE, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      	/*
	 * Increment fried_completed
	 */

	    fried_completed++;
      
	/*
	 * Update progress_percent, and print to stdout.
	 */

	    progress_percent = (100.0 * fried_completed) / fried.get_size();
	    fprintf(stdout, "\r4. Simulating phasescreens: \tIn Progress [%0.3lf %%]", progress_percent);
	    fflush(console);

	/*
	 * Assign new fried parameter, if available, to worker. Else, shutdown. 
	 */

	    if(index_of_fried_in_queue < fried.get_size()){
	    
	    /*
	     * If unprocessed fried parameters are available, send a new one to the worker.
	     */

		MPI_Send(fried.root_ptr + index_of_fried_in_queue, 1, MPI_DOUBLE, status.MPI_SOURCE, mpi_cmds::stayalive, MPI_COMM_WORLD);
	
	    /*
	     * Update process_fried_map, and increment index_of_fried_in_queue)
	     */
	
		process_fried_map[status.MPI_SOURCE] = index_of_fried_in_queue;
		index_of_fried_in_queue++;
      	    }
	    else{
	    
	    /*
	     * If no more fried parameters are available, shutdown processes
	     */

		MPI_Send(nullptr, 0, MPI_CHAR, status.MPI_SOURCE, mpi_cmds::shutdown, MPI_COMM_WORLD);
	    }

	}

    /*
     * !(6) Write phase-screens to output file
     *
     */

	fprintf(console, "\n5. Writing phase to file:");
	if(phase.wr_fits(config::write_phase_to.c_str(), config::output_clobber) != EXIT_SUCCESS){
	    fprintf(console, "\t[Failed]\n");
	}
	else
	    fprintf(console, "\t[Done]\n");


    /*
     * End of workflow
     */

    }else if(process_rank){

    /*
     * Workflow for the worker MPI processes
     */
    
    /*
     * Initialize variables to store dimensions.
     *
     * Name			Type			Purpose
     * dims_phase		std::vector<size_t>	Vector to store the dimensions of a single phase-screen, in pixels.
     * dims_aperture		std::vector<size_t>	Vector to store the dimensions of the aperture, in pixels.
     * dims_phase_per_fried	std::vector<size_t>	Vector to store the dimensions of phase-screens, per fried, in pixels.
     *
     * Additional comments:
     *
     * The size of the phase-screen simulation *should* be much larger than the size of the aperture to avoid \\
     * underestimation of the low-order fluctuations. Therefore, the size of the simulation, in pixels, is appropriately \\
     * scaled. Therefore, dims_phase in this workflow is *not* equal to the dims_phase in the master workflow. 
     */
	sizt_vector dims_phase{sizt(config::phase_size * config::sims_size_x / config::aperture_size), sizt(config::phase_size * config::sims_size_y / config::aperture_size)}; 
        sizt_vector dims_aperture{config::sims_size_x, config::sims_size_y};
	sizt_vector dims_phase_per_fried{config::sims_per_fried, config::sims_size_x, config::sims_size_y};

    /*
     * Initialize phase-screen arrays.
     *
     * Name		Type		Purpose
     * aperture		Array<double>	Array to store the aperture function.
     * phase_per_fried	Array<double>	Array to store the phase-screen simulations, per fried.
     * phase		Array<cmpx>	Array to store a single phase-screen.
     * phase_fourier	Array<cmpx>	Array to store the fourier transformation of a phase-screen.
     *
     * Additional comments:
     *
     * phase and phase_fourier are both re-used for multiple simulations, then clipped \\
     * to the size of the aperture and stored in phase_per_fried.
     */

        Array<double> aperture(dims_aperture);
	Array<double> phase_per_fried(dims_phase_per_fried);
	Array<cmpx>   phase(dims_phase);
	Array<cmpx>   phase_fourier(dims_phase);

    /*
     * Import fft wisdom, if available, and initialize fourier transformation.
     *
     * Name		Type		Purpose
     * forward		fftw_plan	Re-usable FFTW plan for the forward transformation.
     */

        fftw_import_wisdom_from_filename(config::read_fftwisdom_from.c_str());
        fftw_plan forward = fftw_plan_dft_2d(dims_phase[0], dims_phase[1],\
                                             reinterpret_cast<fftw_complex*>(phase_fourier.root_ptr),\
                                             reinterpret_cast<fftw_complex*>(phase.root_ptr),\
                                             FFTW_FORWARD, FFTW_MEASURE);
   
    /*
     * Initialize convenience variables.
     *
     * Name			Type	Purpose
     * fried			double	Variable to store the fried parameter value received from master rank.
     * aperture_radius		double	Variable to store the radius of the aperture, in pixels. Required for clipping.
     * aperture_center_x	sizt	Variable to store the center of the clipping region in x, in pixels.
     * aperture_center_y   	sizt	Variable to store the center of the clipping region in y, in pixels. 
     * phase_center_x		sizt	Variable to store the center of the simulated phase-screen in x, in pixels.
     * phase_center_y		sizt	Variable to store the center of the simulated phase-screen in y, in pixels.
     */

        double fried = 0.0;
	double aperture_radius = config::sims_size_x / (2.0 * config::aperture_sampling_factor);
        sizt aperture_center_x = sizt(config::sims_size_x / 2.0);
        sizt aperture_center_y = sizt(config::sims_size_y / 2.0);
        sizt phase_center_x    = sizt(dims_phase[0] / 2.0);
        sizt phase_center_y    = sizt(dims_phase[1] / 2.0);


    /*
     * Get fried parameter, and aperture function from master.
     */

        MPI_Recv(&fried, 1,  MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(aperture.root_ptr, aperture.get_size(), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        

    /*
     * Enter loop to simulate phase-screens, until shutdown. 
     */

	while(status.MPI_TAG != mpi_cmds::shutdown){
	    
	/*
	 * Simulate <config::sims_per_fried> phase-screens
	 */

            for(sizt ind = 0; ind < config::sims_per_fried; ind++){
             
	        create_phase_screen_fourier_shifted(phase_fourier, fried, config::phase_size);
                fftw_execute_dft(forward, reinterpret_cast<fftw_complex*>(phase_fourier.root_ptr),\
                                          reinterpret_cast<fftw_complex*>(phase.root_ptr));

		for(sizt xs = 0; xs < config::sims_size_x; xs++){
		    for(sizt ys = 0; ys < config::sims_size_y; ys++){
			phase_per_fried(ind, xs, ys) = phase(xs + (phase_center_x - aperture_center_x), ys + (phase_center_y - aperture_center_y)).real();
		    }
		}
            }
	    
	    #ifdef _APODIZE_

	    #endif
	/*
	 * Send the simulation back to the master rank
	 */

	    MPI_Send(phase_per_fried.root_ptr, phase_per_fried.get_size(), MPI_DOUBLE, 0, mpi_pmsg::ready, MPI_COMM_WORLD);
            
	/*
	 * Get the next fried parameter from master.
	 */
	    MPI_Recv(&fried, 1,  MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Finalize();
    return(EXIT_SUCCESS);
}
