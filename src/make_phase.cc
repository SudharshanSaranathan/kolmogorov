#include "mpi.h"
#include "fftw3.h"
#include "config.h"
#include "fitsio.h"
#include "lib_mpi.h"
#include "lib_array.h"
#include "lib_phase.h"

#include <ctime>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <iostream>

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
 * 3. Distribute the fried parameters to MPI processs.
 * 4. Store the simulated phase-screens returned by MPI processs.
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
 * --------------------------------------------
 * Name             Type            Description
 * --------------------------------------------
 * status           MPI_status      MPI status, see MPI documentation.
 * mpi_precision    MPI_Datatype    MPI Datatype, see MPI documentation.
 * process_rank     int             Rank of MPI processes.
 * process_total    int             Store the total number of MPI processes
 * mpi_recv_count   int             Store the count of data received in MPI_Recv, see MPI documentation for explanation.
 * read_status      int             File read status.
 * write_status     int             File write status.
 */
   
    MPI_Status status;
    MPI_Datatype mpi_precision = std::is_same<precision, float>::value == true ? MPI_FLOAT : MPI_DOUBLE;
    int process_rank = 0;
    int processes_total = 0;
    int processes_killed = 0;
    int mpi_recv_count = 0;
    int read_status = 0;
    int write_status = 0;

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

    fprintf(console, "(Info)\tReading configuration:\t");
    fflush (console);
    
    if(config_parse(argv[1]) == EXIT_FAILURE){
	    
        fprintf(console, "[Failed]\n");
        fflush (console);

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    
    }else{

        fprintf(console, "[Done] (%s)\n", argv[1]);
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
     * percent_assigned         float   Percentage of fried parameters assigned to MPI processs.
     * percent_completed        float   Percentage of fried parameters completed by MPI processs.
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

        fprintf(console, "(Info)\tReading file:\t\t");
        fflush (console);
        
        read_status = fried.rd_fits(io_t::read_fried_from.c_str());
        if(read_status != EXIT_SUCCESS){

            fprintf(console, "[Failed][Err code = %d](%s)\n", read_status, io_t::read_fried_from.c_str());
            fflush (console);
            
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);	    

        }
        else{
            
            fprintf(console, "[Done] (%s)\n", io_t::read_fried_from.c_str());
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
    
        const sizt_vector dims_phase{fried.get_size(), sims_t::realizations_per_fried, sims_t::size_x_in_pixels, sims_t::size_y_in_pixels};
        const sizt_vector dims_phase_per_fried{sims_t::realizations_per_fried, sims_t::size_x_in_pixels, sims_t::size_y_in_pixels};
        sizt_vector process_fried_map(fried.get_size() + 1);
    
#ifdef _USE_APERTURE_

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

        fprintf(console, "(Info)\tReading file:\t\t");
        fflush (console);

        read_status = aperture.rd_fits(io_t::read_aperture_function_from.c_str());
        if(read_status != EXIT_SUCCESS){

            fprintf(console, "[Failed][Err code = %d](%s)\n", read_status, io_t::read_aperture_function_from.c_str());
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

        if(dims_aperture[0] != sims_t::size_x_in_pixels && dims_aperture[1] != sims_t::size_y_in_pixels){
	        
            fprintf(console, "[Failed][Expected aperture size = [%lu, %lu]]\n", sims_t::size_x_in_pixels, sims_t::size_y_in_pixels);
            fflush (console);

            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

	    }
        else{

            fprintf(console, "[Done] (%s)\n", io_t::read_aperture_function_from.c_str());
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
     * pid      int     Rank of MPI processes.
     */

        for(int pid = 1; pid < processes_total; pid++){

        /* --------------------------------------------------
         * if rank > number of fried parameters, kill MPI process.
         * -------------------------------------------------- 
         */

            if(pid > int(fried.get_size())){

                MPI_Send(fried[0], 1, mpi_precision, pid, mpi_cmds::kill, MPI_COMM_WORLD);

#ifdef _USE_APERTURE_
                
                MPI_Send(aperture[0], aperture.get_size(), mpi_precision, pid, mpi_cmds::kill, MPI_COMM_WORLD);

#endif
                
            /* -----------------------------------------
             * Decrement number of MPI processes in use.
             * -----------------------------------------
             */
		
                processes_killed++;          

            }

        /* --------------------------------------------------------------------------------------------
         * if rank <= number of fried parameters, send fried parameter and aperture function to MPI process.
         * --------------------------------------------------------------------------------------------
         */
            
            else{

                if(fried[index_of_fried_in_queue] != nullptr){

                    MPI_Send(fried[index_of_fried_in_queue], 1, mpi_precision, pid, mpi_cmds::task, MPI_COMM_WORLD);

                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);

                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                }

#ifdef _USE_APERTURE_

                if(aperture[0] != nullptr){
                    
                    MPI_Send(aperture[0], aperture.get_size(), mpi_precision, pid, mpi_cmds::task, MPI_COMM_WORLD);

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

                process_fried_map[pid] = index_of_fried_in_queue;

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
            
        processes_total -= processes_killed;

    /*
     * Array declaration:
     * ------------------------------------
     * Name     Type                Description
     * ------------------------------------
     * phase	Array<precision>	Phase-screens array.
     */

        Array<precision> phase(dims_phase);

    /* ------------------------------------------------
     * !(3) Distribute the fried parameters to MPI processs.
     * !(4) Store the simulated phase-screens returned by MPI processs.
     * !(5) Repeat steps 3-4 for all fried parameters.
     * -----------------------------------------------
     */

        while(fried_completed < fried.get_size()){
        	  
        /* --------------------------------------------------------------------------------
         * Wait for a MPI processes to ping master. If pinged, get MPI process information.
         * --------------------------------------------------------------------------------
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
         * Get index of fried parameter processed by MPI process.
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

        /* ---------------------------------------------------------------------
         * Assign new fried parameter to MPI process if available, else, kill MPI process.
         * ---------------------------------------------------------------------
         */

            if(index_of_fried_in_queue < fried.get_size()){
	    
            /* ----------------------------------------------------------------------------
             * If unprocessed fried parameters are available, send a new one to the MPI process.
             * ---------------------------------------------------------------------------- 
             */

                if(fried[index_of_fried_in_queue] != nullptr){

                    MPI_Send(fried[index_of_fried_in_queue], 1, mpi_precision, status.MPI_SOURCE, mpi_cmds::task, MPI_COMM_WORLD);
	
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
      	    
            }else{
    
            /* ---------------------------------------------------------
             * No more fried parameters to process, so kill MPI workers.
             * ---------------------------------------------------------
             */

                MPI_Send(fried[0], 1, mpi_precision, status.MPI_SOURCE, mpi_cmds::kill, MPI_COMM_WORLD);

            }
        }

    /* --------------------------------------------
     * !(6) Save simulations to disk, if requested.
     * --------------------------------------------
     */

        if(io_t::save){


            fprintf(console, "\n(Info)\tWriting to file:\t");
            fflush (console);

            write_status = phase.wr_fits(io_t::write_phase_to.c_str(), io_t::clobber);
            if(write_status != EXIT_SUCCESS){
            
                fprintf(console, "[Failed][Err code = %d](%s)\n", write_status, io_t::write_phase_to.c_str());
                fflush (console);

            }else{

                fprintf(console, "[Done] (%s)\n", io_t::write_phase_to.c_str());
                fflush (console);

            }
        }

    /*
     * --------------------------------
     * End of workflow for MPI rank = 0
     * --------------------------------
     */

    }
    
/*
 * -------------------------
 * Workflow for MPI rank > 0
 * -------------------------
 */    
    
    else if(process_rank){
    
    /*
     * Vector declaration:
     * ----------------------------------------------------
     * Name                     Type            Description
     * ----------------------------------------------------
     * dims_phase_per_fried     sizt_vector     Dimensions of the cropped phase-screens in pixels, per fried.
     * dims_aperture            sizt_vector     Dimensions of the aperture function, in pixels.
     * dims_phase               sizt_vector     Dimensions of a single simulated phase-screen, in pixels.
     *
     * --------------------
     * Additional comments:
     * --------------------
     * The size of the simulations, in pixels, *should* be much larger than the size of the aperture to avoid
     * underestimation of the low-orders. Therefore dims_phase in this workflow is *not* equal to dims_phase
     * in the previous workflow. 
     */

        const sizt_vector dims_phase_per_fried{sims_t::realizations_per_fried, sims_t::size_x_in_pixels, sims_t::size_y_in_pixels};
        const sizt_vector dims_aperture{sims_t::size_x_in_pixels, sims_t::size_y_in_pixels};
        const sizt_vector dims_phase{sizt(sims_t::size_in_meters * sims_t::size_x_in_pixels * aperture_t::sampling_factor / aperture_t::size),\
                                     sizt(sims_t::size_in_meters * sims_t::size_y_in_pixels * aperture_t::sampling_factor / aperture_t::size)};

    /*
     * Vector declaration
     * --------------------------------------------
     * Name             Type            Description
     * --------------------------------------------
     * dims_crop_start  sizt_vector     The starting coordinate for cropping the simulations.
     */

        const sizt_vector dims_crop_start{(dims_phase[0] - dims_aperture[0]) / 2, (dims_phase[1] - dims_aperture[1]) / 2};

    /*
     * Array declaration:
     * ------------------------------------------------
     * Name             Type                Description
     * ------------------------------------------------
     * phase            Array<cmpx>         Simulated phase-screen.
     * phase_fourier	Array<cmpx>         Fourier of simulated phase-screen.
     * phase_per_fried  Array<precision>    Phase-screen simulations per fried, cropped.
     * aperture         Array<precision>    Aperture function.
     *
     * --------------------
     * Additional comments:
     * --------------------
     * phase and phase_fourier are re-used over the requested number of realizations, 
     * cropped to the dimensions of the aperture and stored in phase_per_fried. 
     * phase_per_fried is then sent to MPI rank = 0.
     */

        Array<cmpx>      phase(dims_phase);
        Array<cmpx>      phase_fourier(dims_phase);
        Array<precision> phase_per_fried(dims_phase_per_fried);
        Array<precision> aperture(dims_aperture);

    /* -------------------------------
     * Import fft wisdom if available.
     * -------------------------------
     */

        fftw_import_wisdom_from_filename(io_t::read_fft_phase_wisdom_from.c_str());

    /*
     * Variable declaration:
     * --------------------------------
     * Name     Type        Description
     * --------------------------------
     * forward  fftw_plan   Re-usable FFTW plan for the forward transformation.
     */

        fftw_plan reverse = fftw_plan_dft_2d(dims_phase[0], dims_phase[1],\
                                             reinterpret_cast<fftw_complex*>(phase_fourier[0]),\
                                             reinterpret_cast<fftw_complex*>(phase[0]),\
                                             FFTW_BACKWARD, FFTW_MEASURE);
   
    /*
     * Variable declaration:.
     * --------------------------------------------
     * Name                 Type        Description
     * --------------------------------------------
     * fried                precision   Fried parameter value received from master rank.
     * aperture_total       precision   Area of the aperture.
     * aperture_center_x    sizt        Center of the clipping region in x, in pixels.
     * aperture_center_y    sizt        Center of the clipping region in y, in pixels. 
     * phase_center_x       sizt        Center of the simulated phase-screen in x, in pixels.
     * phase_center_y       sizt        Center of the simulated phase-screen in y, in pixels.
     */

        precision fried = 0.0;

        sizt aperture_center_x = sizt(sims_t::size_x_in_pixels / 2.0);
        sizt aperture_center_y = sizt(sims_t::size_y_in_pixels / 2.0);
        sizt phase_center_x    = sizt(dims_phase[0] / 2.0);
        sizt phase_center_y    = sizt(dims_phase[1] / 2.0);

    /* --------------------------------
     * Get fried parameter from master.
     * --------------------------------
     */

        MPI_Recv(&fried, 1,  mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

#ifdef _USE_APERTURE_

    /* ------------------------------------------------
     * If aperture function available, get from master.
     * ------------------------------------------------
     */

        MPI_Recv(aperture[0], aperture.get_size(),  mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    /*
     * Variable declaration:
     * ----------------------------------------
     * Name             Type        Description
     * ----------------------------------------
     * aperture_total   precision   Area of the aperture.
     */

        precision aperture_total  = aperture.get_total();

#endif

    /* ---------------------------------------------------
     * Enter loop to simulate phase-screens, until killed.
     * ---------------------------------------------------
     */

        while(status.MPI_TAG != mpi_cmds::kill){

            for(sizt ind = 0; ind < sims_t::realizations_per_fried; ind++){
            	    
                precision phase_piston = 0.0;

            /* -------------------------------
             * Simulate a single phase-screen.
             * -------------------------------
             */

                make_phase_screen_fourier_shifted(phase_fourier, fried, sims_t::size_in_meters * aperture_t::sampling_factor);
                fftw_execute_dft(reverse, reinterpret_cast<fftw_complex*>(phase_fourier[0]), reinterpret_cast<fftw_complex*>(phase[0]));

#ifdef _USE_APERTURE_
	
            /* ---------------------------------------------------
             * If aperture available, clip simulation to aperture.
             * ---------------------------------------------------
             */

                for(sizt xs = 0; xs < sims_t::size_x_in_pixels; xs++){
                    for(sizt ys = 0; ys < sims_t::size_y_in_pixels; ys++){

                        phase_per_fried(ind, xs, ys) = aperture(xs, ys) * static_cast<precision>(phase(xs + phase_center_x - aperture_center_x, ys + phase_center_y - aperture_center_y).real());
                        phase_piston += phase_per_fried(ind, xs, ys) / aperture_total;
                    
                    }
                }
	    
            /* ------------------------------------------------------
             * Subtract phase-screen mean over aperture a.k.a piston.
             * ------------------------------------------------------
             */
                
                for(sizt xs = 0; xs < sims_t::size_x_in_pixels; xs++){
                    for(sizt ys = 0; ys < sims_t::size_y_in_pixels; ys++){
                        phase_per_fried(ind, xs, ys) -= aperture(xs, ys) * phase_piston;
                    }
                }

#else
	
            /* -------------------------------------------------------------
             * If aperture not available, clip simulation to requested size.
             * -------------------------------------------------------------
             */

                for(sizt xs = 0; xs < sims_t::size_x_in_pixels; xs++){
                    for(sizt ys = 0; ys < sims_t::size_y_in_pixels; ys++){
                        phase_per_fried(ind, xs, ys) = static_cast<precision>(phase(xs + (phase_center_x - aperture_center_x), ys + (phase_center_y - aperture_center_y)).real());
                        phase_piston += phase_per_fried(ind, xs, ys);
                    }
                }

            /* ----------------------------------------
             * Subtract phase-screen mean a.k.a piston.
             * ----------------------------------------
             */

                phase_piston /= (sims_t::size_x_in_pixels * sims_t::size_y_in_pixels);
                for(sizt xs = 0; xs < sims_t::size_x_in_pixels; xs++){
                    for(sizt ys = 0; ys < sims_t::size_y_in_pixels; ys++){
                        phase_per_fried(ind, xs, ys) -= phase_piston;
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

    /* -------------------------
     * Write FFT wisdom to file.
     * -------------------------
     */
            
        fftw_export_wisdom_to_filename(io_t::read_fft_phase_wisdom_from.c_str());
        fftw_destroy_plan(reverse);
        fftw_cleanup();

    }

    MPI_Finalize();
    return(EXIT_SUCCESS);
}
