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

#define _RESIDUAL_
#define _APERTURE_

/* ------------
 * Description:
 * ------------
 * This program simulates phase-screens at the exit pupil of a telescope that have been \\
 * degraded by atmospheric turbulence. The phase-screens are computed as the fourier transform \\
 * of a complex array of spatial frequencies. The amplitude of the complex array follows the \\
 * Kolmogorov-Obukhov power law, and the phase array consists of zero-mean, unit-variance Gaussian \\
 * random numbers. 
 *
 * ------
 * Usage:
 * ------
 * mpiexec -np <cores> ./make_phase <config_file>
 *
 * -------
 * Inputs:
 * -------
 * See config.h for a detailed explanation of the inputs to the program.
 *
 * --------
 * Outputs:
 * --------
 * See config.h for a detailed explanation of the output of the program.
 *
 * --------------
 * Program logic:
 * -------------- 
 * 1. Read and parse the config file.
 * 2. Read fried parameters from file.
 * 3. Distribute the fried parameters to workers.
 * 4. Store the simulated phase-screens returned by workers.
 * 5. Repeat steps 3-4 for all fried parameters.
 * 6. Save simulations to disk.
 *
 * -----------------------
 * Additional information:
 * -----------------------
 * Each comment block has a title that explains the succeeding code. 
 * Titles starting with !, followed by a number n indicate a block handling step n.
 */

int main(int argc, char *argv[]){

/*
 * Variable declaration:
 * ----------------------------------------
 * Name             Type        Description
 * ----------------------------------------
 * status           MPI_status  See MPI documentation.
 * process_rank     int         Rank of MPI processes.
 * process_total    int         Store the total number of MPI processes
 * mpi_recv_count   int         Store the count of data received in MPI_Recv, see MPI documentation for explanation.
 * read_status      int         File read status.
 * write_status     int         File write status.
 * mpi_precision    int         MPI_FLOAT or MPI_DOUBLE.
 */
   
    MPI_Status status;
    int process_rank = 0;
    int processes_total = 0;
    int mpi_recv_count = 0;
    int read_status = 0;
    int write_status = 0;
    int mpi_precision = std::is_same<precision, float>::value == true ? MPI_FLOAT : MPI_DOUBLE;

/* -------------------------
 * Initialize MPI framework.
 * ------------------------- 
 */
   
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processes_total);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

/* ------------------------------------------------------
 * Only the master MPI process (rank 0) prints to stdout.
 * ------------------------------------------------------
 */

    FILE *console   = process_rank == 0 ? stdout : fopen("/dev/null","wb");
    fprintf(console, "------------------------------------------------------\n");
    fprintf(console, "- Turbulence-degraded phasescreen simulation program -\n");
    fprintf(console, "------------------------------------------------------\n");

/* ------------------------------------
 * !(1) Read and parse the config file.
 * ------------------------------------
 */

    if(argc < 2){

        fprintf(console, "(Error)\tExpected configuration file, calling MPI_Abort()\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
   
    }

    fprintf(console, "(Info)\tReading configuration:\t[%s, ", argv[1]);
    fflush (console);
    
    if(config_parse(argv[1]) == EXIT_FAILURE){
	    
        fprintf(console, "Failed]\n");
        fflush (console);

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    
    }else{

        fprintf(console, "Done]\n");
        fflush (console);

    }
    
/*
 * -------------------------
 * Workflow for master rank.
 * -------------------------
 */

    if(process_rank == 0){
	
    /*
     * Variable declaration:
     * ------------------------------------------------
     * Name                     Type    Description
     * ------------------------------------------------
     * index_of_fried_in_queue  ulng    Index of the next fried parameter.
     * fried_completed          ulng    Number of fried parameters processed.
     * percent_assigned         float   Percentage of fried parameters assigned to workers.
     * percent_completed        float   Percentage of fried parameters completed by workers.
     */

        ulng  index_of_fried_in_queue  = 0;
        ulng  fried_completed          = 0;
        float percent_assigned         = 0.0;
        float percent_completed        = 0.0;

    /*
     * Array declaration:
     * ----------------------------------------
     * Name		Type		        Description
     * ----------------------------------------
     * fried	Array<precision>	Fried parameters array, see "lib_array.h" for datatype.
     */
    
        Array<precision> fried;

    /* -------------------------------------
     * !(2) Read fried parameters from file.
     * -------------------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::read_fried_from.c_str());
        fflush (console);
        
        read_status = fried.rd_fits(config::read_fried_from.c_str());
        if(read_status != EXIT_SUCCESS){

            fprintf(console, "Failed with err code: %d]\n", read_status);
            fflush (console);
            
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);	    

        }
        else{
            
            fprintf(console, "Done]\n");
            fflush (console);
        
        }

    /*
    * Vector declaration:
    * -------------------------------------------------
    * Name			        Type			Description
    * -------------------------------------------------
    * dims_phase            sizt_vector     Dimensions of phase-screens, in pixels.
    * dims_phase_per_fried  sizt_vector     Dimensions of phase-screens, per fried, in pixels.
    * process_fried_map     sizt_vector     Map of which MPI process is working on which fried parameter.
    */
    
        const sizt_vector dims_phase{fried.get_size(), config::sims_per_fried, config::sims_size_x, config::sims_size_y};
        const sizt_vector dims_phase_per_fried{config::sims_per_fried, config::sims_size_x, config::sims_size_y};
        sizt_vector process_fried_map(fried.get_size() + 1);
    
#ifdef _APERTURE_

    /*
     * Array declaration:
     * -------------------------------------------
     * Name         Type                Description
     * --------------------------------------------
     * aperture     Array<precision>    Aperture function array, see "lib_array.h" for datatype.
     */
	
        Array<precision> aperture;

    /* -----------------------------------------------
     * If aperture function available, read from file.
     * -----------------------------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::read_aperture_function_from.c_str());
        fflush (console);

        read_status = aperture.rd_fits(config::read_aperture_function_from.c_str());
        if(read_status != EXIT_SUCCESS){

            fprintf(console, "Failed with err code: %d]\n", read_status);
            fflush (console);
            
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

	    }

    /*
     * Vector declaration:
     * --------------------------------------------
     * Name             Type            Description
     * --------------------------------------------
     * dims_aperture    sizt_vector     Dimensions of the aperture, in pixels.
     */
        
        const sizt_vector dims_aperture = aperture.get_dims();

    /* ----------------------------------------------------------------------------
     * Check that dimensions of the aperture match values specified in config file.
     * ----------------------------------------------------------------------------
     */

        if(dims_aperture[0] != config::sims_size_x && dims_aperture[1] != config::sims_size_y){
	        
            fprintf(console, "Failed, expected aperture with size [%ud %ud]]\n", config::sims_size_x, config::sims_size_y);
            fflush (console);

            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

	    }
        else{

            fprintf(console, "Done]\n");
            fflush (console);

        }

#endif
        
        percent_assigned  = (100.0 * index_of_fried_in_queue) / fried.get_size();
        percent_completed = (100.0 * fried_completed) / fried.get_size();
        
        fprintf(console, "\r(Info)\tSimulating phases:\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed);
        fflush (console);
    
    /*
     * Variable declaration:
     * ----------------------------
     * Name     Type    Description
     * ----------------------------
     * id       int     Rank of MPI processes.
     */

        for(int id = 1; id < processes_total; id++){

        /* ------------------------------------------------------
         * if rank < number of fried parameters. Shutdown worker.
         * ------------------------------------------------------ 
         */

            if(id > int(fried.get_size())){

                if(fried[0] != nullptr){

                    MPI_Send(fried[0], 1, mpi_precision, id, mpi_cmds::shutdown, MPI_COMM_WORLD);

                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);

                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                }

#ifdef _APERTURE_
                
                if(aperture[0] != nullptr){

                    MPI_Send(aperture[0], aperture.get_size(), mpi_precision, id, mpi_cmds::shutdown, MPI_COMM_WORLD);

                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);

                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                }

#endif
                
            /* -------------------------------------
             * Decrement number of processes in use.
             * -------------------------------------
             */
		
                processes_total--;          

            }
            
        /* --------------------------------------------------------------------------------------------
         * if rank >= number of fried parameters, send fried parameter and aperture function to worker.
         * --------------------------------------------------------------------------------------------
         */
            
            else{

                if(fried[index_of_fried_in_queue] != nullptr){

                    MPI_Send(fried[index_of_fried_in_queue], 1, mpi_precision, id, mpi_cmds::stayalive, MPI_COMM_WORLD);

                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);

                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                }

#ifdef _APERTURE_

                if(aperture[0] != nullptr){
                    
                    MPI_Send(aperture[0], aperture.get_size(), mpi_precision, id, mpi_cmds::stayalive, MPI_COMM_WORLD);

                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);

                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                }

#endif

            /* -------------------------------------------------------
             * Store index_of_fried_in_queue in process_fried_map[id].
             * -------------------------------------------------------
             */

                process_fried_map[id] = index_of_fried_in_queue;

            /* ----------------------------------
             * Increment index_of_fried_in_queue.
             * ----------------------------------
	         */

                index_of_fried_in_queue++;

            /* ------------------------------------
             * Update and display percent_assigned.
             * ------------------------------------
             */

                percent_assigned  = (100.0 * index_of_fried_in_queue) / fried.get_size();
       
                fprintf(console, "\r(Info)\tSimulating phases:\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
                fflush (console);

            }
        }

    /*
     * Array declaration:
     * ------------------------------------
     * Name     Type                Description
     * ------------------------------------
     * phase	Array<precision>	Phase-screens array.
     */

        Array<precision> phase(dims_phase);

    /* ------------------------------------------------
     * !(3) Distribute the fried parameters to workers.
     * !(4) Store the simulated phase-screens returned by workers.
     * !(5) Repeat steps 3-4 for all fried parameters.
     * -----------------------------------------------
     */

        while(fried_completed < fried.get_size()){
        	  
        /* ----------------------------------------------------------------------------
         * Wait for a worker that is ready. If found, get and store worker information.
         * ----------------------------------------------------------------------------
         */	
	
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, mpi_precision, &mpi_recv_count);

        /*
         * Variable declaration:
         *---------------------------------
         * Name         Type    Description
         * --------------------------------
         * fried_index  sizt    Index of simulated fried parameter.
         */

        /* -------------------------------------------------
         * Get index of fried parameter processed by worker.
         * -------------------------------------------------
         */
            
            sizt fried_index = process_fried_map[status.MPI_SOURCE];

        /* -----------------------------------------------------
         * Get data, and store in phase at the correct location.
         * -----------------------------------------------------
         */
         
            if(phase[fried_index] != nullptr){

                MPI_Recv(phase[fried_index], sizeof_vector(dims_phase_per_fried), mpi_precision, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            }else{
                
                fprintf(console, "\n(Error)\tNull buffer, calling MPI_Abort()\n");
                fflush (console);
       
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

            }

        /* --------------------------
         * Increment fried_completed.
         * --------------------------
         */

            fried_completed++;
      
        /* -------------------------------------
         * Update and display percent_completed.
         * ------------------------------------- 
         */

            percent_completed = (100.0 * fried_completed) / fried.get_size();
	        
            fprintf(console, "\r(Info)\tSimulating phases:\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed);
            fflush (console);

        /* --------------------------------------------------------------------
         * Assign new fried parameter, if available, to worker, else, shutdown. 
         * --------------------------------------------------------------------
         */

            if(index_of_fried_in_queue < fried.get_size()){
	    
            /* ----------------------------------------------------------------------------
             * If unprocessed fried parameters are available, send a new one to the worker.
             * ---------------------------------------------------------------------------- 
             */

                if(fried[index_of_fried_in_queue] != nullptr){

                    MPI_Send(fried[index_of_fried_in_queue], 1, mpi_precision, status.MPI_SOURCE, mpi_cmds::stayalive, MPI_COMM_WORLD);
	
                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);

                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                }

            /* -------------------------
             * Update process_fried_map.
             * -------------------------
             */
	
                process_fried_map[status.MPI_SOURCE] = index_of_fried_in_queue;

            /* ----------------------------------
             * Increment index_of_fried_in_queue.
             * ----------------------------------
             */

                index_of_fried_in_queue++;

            /* ------------------------------------
             * Update and display percent_assigned.
             * ------------------------------------
             */

                percent_assigned  = (100.0 * index_of_fried_in_queue) / fried.get_size();
                
                fprintf(console, "\r(Info)\tSimulating phases:\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
                fflush (console);
      	    
            }
	        
        /* --------------------------------------------------------------
         * If no more fried parameters are available, shutdown processes.
         * --------------------------------------------------------------
         */

            else{

                MPI_Send(nullptr, 0, MPI_CHAR, status.MPI_SOURCE, mpi_cmds::shutdown, MPI_COMM_WORLD);
	        
            /* --------------------------
             * Decrement processes_total;
             * --------------------------
             */

                processes_total--;

            }
        }

    /* ------------------------------ 
     * !(6) Save simulations to disk.
     * ------------------------------
     */

        fprintf(console, "\n(Info)\tWriting to file:\t[%s, ", config::write_phase_to.c_str());
        fflush (console);

        write_status = phase.wr_fits(config::write_phase_to.c_str(), config::output_clobber);
        if(write_status != EXIT_SUCCESS){
            
            fprintf(console, "Failed with err code: %d]\n", write_status);
            fflush (console);

        }else{

            fprintf(console, "Done]\n");
            fflush (console);

        }

    /*
     * -------------------------------
     * End of workflow for master rank
     * -------------------------------
     */

    }
    
/*
 * ------------------------
 * Workflow for worker rank
 * ------------------------
 */    
    
    else if(process_rank){
    
    /*
     * Vector declaration:
     * ----------------------------------------------------
     * Name                     Type            Description
     * ----------------------------------------------------
     * dims_phase               sizt_vector     Dimensions of single phase-screen, in pixels.
     * dims_aperture            sizt_vector     Dimensions of the aperture function, in pixels.
     * dims_phase_per_fried     sizt_vector     Dimensions of phase-screens, per fried, in pixels.
     *
     * --------------------
     * Additional comments:
     * --------------------
     * The size of the phase-screen simulation *should* be much larger than the size of the aperture to avoid \\
     * underestimation of the low-order fluctuations. Therefore, the size of the simulation, in pixels, is appropriately \\
     * scaled. Therefore, dims_phase in this workflow is *not* equal to the dims_phase in the master workflow. 
     */

        const sizt_vector dims_phase{sizt(config::phase_size * config::sims_size_x / config::aperture_size), sizt(config::phase_size * config::sims_size_y / config::aperture_size)};
        const sizt_vector dims_aperture{config::sims_size_x, config::sims_size_y};
        const sizt_vector dims_phase_per_fried{config::sims_per_fried, config::sims_size_x, config::sims_size_y};

    /*
     * Array declaration:
     * ------------------------------------------------
     * Name             Type                Description
     * ------------------------------------------------
     * phase            Array<cmpx>         Single phase-screen array.
     * phase_fourier	Array<cmpx>         Single phase-screen fourier array.
     * phase_per_fried  Array<precision>    Phase-screens array, per fried.
     * aperture         Array<precision>    Aperture function array.
     *
     * --------------------
     * Additional comments:
     * --------------------
     * phase and phase_fourier are both re-used for multiple simulations, then clipped \\
     * to the size of the aperture and stored in phase_per_fried.
     */

        Array<cmpx>      phase(dims_phase);
        Array<cmpx>      phase_fourier(dims_phase);
        Array<precision> phase_per_fried(dims_phase_per_fried);
        Array<precision> aperture(dims_aperture);

    /* -----------------------------------------------------------------------
     * Import fft wisdom, if available, and initialize fourier transformation.
     * -----------------------------------------------------------------------
     */

        fftw_import_wisdom_from_filename(config::read_fftwisdom_from.c_str());

    /*
     * Variable declaration:
     * --------------------------------
     * Name     Type        Description
     * --------------------------------
     * forward  fftw_plan   Re-usable FFTW plan for the forward transformation.
     */

        fftw_plan forward = fftw_plan_dft_2d(dims_phase[0], dims_phase[1],\
                                             reinterpret_cast<fftw_complex*>(phase_fourier[0]),\
                                             reinterpret_cast<fftw_complex*>(phase[0]),\
                                             FFTW_FORWARD, FFTW_MEASURE);
   
    /*
     * Variable declaration:.
     * --------------------------------------------
     * Name                 Type        Description
     * --------------------------------------------
     * fried                precision   Fried parameter value received from master rank.
     * aperture_total       precision   Area of the aperture;
     * aperture_center_x    sizt        Center of the clipping region in x, in pixels.
     * aperture_center_y    sizt        Center of the clipping region in y, in pixels. 
     * phase_center_x       sizt        Center of the simulated phase-screen in x, in pixels.
     * phase_center_y       sizt        Center of the simulated phase-screen in y, in pixels.
     */

        precision fried = 0.0;
        precision aperture_total  = 0.0;

        sizt aperture_center_x = sizt(config::sims_size_x / 2.0);
        sizt aperture_center_y = sizt(config::sims_size_y / 2.0);
        sizt phase_center_x    = sizt(dims_phase[0] / 2.0);
        sizt phase_center_y    = sizt(dims_phase[1] / 2.0);

    /* --------------------------------
     * Get fried parameter from master.
     * --------------------------------
     */

        MPI_Recv(&fried, 1,  mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

#ifdef _APERTURE_

    /* ------------------------------------------------
     * If aperture function available, get from master.
     * ------------------------------------------------
     */

        MPI_Recv(aperture[0], aperture.get_size(),  mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        aperture_total = aperture.get_total();

#endif

    /* -----------------------------------------------------
     * Enter loop to simulate phase-screens, until shutdown.
     * -----------------------------------------------------
     */

        while(status.MPI_TAG != mpi_cmds::shutdown){

            for(sizt ind = 0; ind < config::sims_per_fried; ind++){
            	    
    	    /* -------------------------------
	         * Simulate a single phase-screen.
             * -------------------------------
	        */
             
                make_phase_screen_fourier_shifted(phase_fourier, fried, config::phase_size);
                fftw_execute_dft(forward, reinterpret_cast<fftw_complex*>(phase_fourier[0]),\
                                          reinterpret_cast<fftw_complex*>(phase[0]));

#ifdef _APERTURE_
	
	        /* ---------------------------------------------------
	         * If aperture available, clip simulation to aperture.
             * ---------------------------------------------------
	         */

		        precision phase_piston = 0.0;
		        for(sizt xs = 0; xs < config::sims_size_x; xs++){
		            for(sizt ys = 0; ys < config::sims_size_y; ys++){
			            phase_per_fried(ind, xs, ys) = aperture(xs, ys) * static_cast<precision>(phase(xs + (phase_center_x - aperture_center_x), ys + (phase_center_y - aperture_center_y)).real());
			            phase_piston += aperture(xs, ys) * phase_per_fried(ind, xs, ys);
		            }
		        }
	    
	        /* ------------------------------------------------------
	         * Subtract phase-screen mean over aperture a.k.a piston.
             * ------------------------------------------------------
	         */

		        phase_piston /= aperture_total;
		        for(sizt xs = 0; xs < config::sims_size_x; xs++){
		            for(sizt ys = 0; ys < config::sims_size_y; ys++){
		        	    phase_per_fried(ind, xs, ys) -= aperture(xs, ys) * phase_piston;
		            }
		        }

#else
	
	        /* -------------------------------------------------------------
	         * If aperture not available, clip simulation to requested size.
             * -------------------------------------------------------------
	         */

		        for(sizt xs = 0; xs < config::sims_size_x; xs++){
		            for(sizt ys = 0; ys < config::sims_size_y; ys++){
		    	        phase_per_fried(ind, xs, ys) = static_cast<precision>(phase(xs + (phase_center_x - aperture_center_x), ys + (phase_center_y - aperture_center_y)).real());
		            }
		        }

#endif

            }
	    
	    /* ----------------------------------
	     * Send phase-screens to master rank.
         * ----------------------------------
	     */

            if(phase_per_fried[0] != nullptr){

	            MPI_Send(phase_per_fried[0], phase_per_fried.get_size(), mpi_precision, 0, mpi_pmsg::ready, MPI_COMM_WORLD);

            }else{

                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
               
            }

	    /* -------------------------------------
	     * Get next fried parameter from master.
         * -------------------------------------
	     */

	        MPI_Recv(&fried, 1,  mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        }

        /* ------------------------
         * Write FFT wisdom to file
         * ------------------------
         */
            
            fftw_export_wisdom_to_filename(config::read_fftwisdom_from.c_str());
    }

    MPI_Finalize();
    return(EXIT_SUCCESS);
}
