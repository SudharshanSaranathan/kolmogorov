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

#define _APERTURE_

/* ------------
 * Description:
 * ------------
 * This program computes the Point Spread Function (PSF) corresponding to phase-screen residuals.
 * The PSF is computed as the forward fourier transform of the pupil function that is defined as:
 *
 *      Pupil_function(x, y) = aperture_function(x, y) * exp(i * phase(x, y))
 *
 * where 'phase' denotes an individual phase-screen.
 *
 * ------
 * Usage:
 * ------
 * mpiexec -np <cores> ./make_psf <config_file>
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
 * 2. Read phase-screen residuals from file.
 * 3. Read aperture function from file.
 * 4. Distribute residual phase-screens to workers.
 * 5. Store PSFs returned by workers.
 * 6. Repeat steps 4-5 for all residual phase-screens.
 * 7. Save PSFs to disk.
 *
 * -----------------------
 * Additional information:
 * -----------------------
 * Each comment block has a title that explains the succeeding code. 
 * Titles starting with !, followed by a number n indicate a block handling step n.
 */

int main(int argc, char *argv[]){

/*
 *  Variable declaration:
 *  -------------------------------------------
 *  Name		    Type            Description
 *  -------------------------------------------
 *  status          MPI_status      See MPI documentation.
 *  mpi_precision   MPI_Datatype    MPI_FLOAT or MPI_DOUBLE.
 *  process_rank    int             Rank of MPI processes.
 *  process_total   int             Store the total number of MPI processes
 *  mpi_recv_count  int             Store the count of data received in MPI_Recv, see MPI documentation for explanation.
 *  read_status     int             File read status.
 *  write_status    int             File write status.
 */
   
    MPI_Status status;
    MPI_Datatype mpi_precision = std::is_same<precision, float>::value == true ? MPI_FLOAT : MPI_DOUBLE;
    int process_rank = 0;
    int processes_total = 0;
    int mpi_recv_count = 0;
    int read_status = 0;
    int write_status = 0;

/* --------------
 * Initialize MPI
 * --------------
 */
   
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processes_total);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

/* -----------------------------------------------------------
 * Only the master MPI process - rank zero - prints to stdout.
 * -----------------------------------------------------------
 */

    FILE *console   = process_rank == 0 ? stdout : fopen("/dev/null","wb");
    fprintf(console, "------------------------------------------------------\n");
    fprintf(console, "- Phase-screen PSFs computation program -\n");
    fprintf(console, "------------------------------------------------------\n");

/* ------------------------------------
 * !(1) Read and parse the config file.
 * ------------------------------------
 */

    if(argc < 2){
        fprintf(console, "(Error)\tExpected configuration file, aborting!\n");
        fflush (console);
        
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    fprintf(console, "(Info)\tReading configuration:\t");
    fflush (console);
    
    if(config_parse(argv[1]) == EXIT_FAILURE){
        fprintf(console, "[Failed]\n");
        fflush (console);

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

    }else{
	    
        fprintf(console, "[Done]\n");
        fflush (console);

    }

/* 
 * ------------------------
 * Workflow for master rank
 * ------------------------
 */

    if(process_rank == 0){

    /* 
     * Variable declaration.
     * --------------------------------------------
     * Name                     Type    Description
     * --------------------------------------------
     * index_of_fried_in_queue  ulng    Index of the next fried parameter.
     * fried_completed          ulng    Number of fried parameters processed.
     * percent_assigned         float   Percentage of fried assigned.
     * percent_completed        float   Percentage of fried completed.
     */

        ulng  index_of_fried_in_queue = 0;
        ulng  fried_completed         = 0;
        float percent_assigned        = 0.0;
        float percent_completed       = 0.0;

    /*
     * Array declaration.
     * --------------------------------------------
     * Name         Type                Description
     * --------------------------------------------
     * residual     Array<precision>    Phase-screen residuals, see 'lib_array.h' for datatype.
     * aperture     Array<precision>    Aperture function, see 'lib_array.h' for datatype.
     */

        Array<precision> residual;
        Array<precision> aperture;

    /* -------------------------------------------
     * !(2) Read residual phase-screens from file.
     * -------------------------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::write_residual_to.c_str());
        fflush (console);

        read_status = residual.rd_fits(config::write_residual_to.c_str());
        if(read_status != EXIT_SUCCESS){
            
            fprintf(console, "Failed with err code: %d]\n", read_status);
            fflush (console);

            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        
        }else{
            
            fprintf(console, "Done]\n");
            fflush (console);

        }

    /* --------------------------------------
     * !(3) Read aperture function from file.
     * --------------------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::read_aperture_function_from.c_str());
        fflush (console);

        read_status = aperture.rd_fits(config::read_aperture_function_from.c_str());
        if(read_status != EXIT_SUCCESS){
            
            fprintf(console, "Failed with err code: %d]\n", read_status);
            fflush (console);

            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        
        }else{

            fprintf(console, "Done]\n");
            fflush (console);

        }

    /*
     * Vector declaration.
     * ----------------------------------------------------
     * Name                     Type            Description
     * ----------------------------------------------------
     * dims_residual            sizt_vector     Dimensions of phase-screen residuals.
     * dims_residual_per_fried  sizt_vector     Dimensions of phase-screen residuals, per fried.
     * dims_psf                 sizt_vector     Dimensions of PSFs.
     * dims_psf_per_fried       sizt_vector     Dimensions of PSFs, per_fried.
     * process_fried_map        sizt_vector     Map of which process is handling which fried index.
     */

        const sizt_vector dims_residual = residual.get_dims();
        const sizt_vector dims_residual_per_fried(dims_residual.begin() + 1, dims_residual.end());
        const sizt_vector dims_psf{dims_residual[0], dims_residual[1], 2 * dims_residual[2] - 1, 2 * dims_residual[3] - 1};
        const sizt_vector dims_psf_per_fried(dims_psf.begin() + 1, dims_psf.end());
        sizt_vector process_fried_map(dims_residual[0] + 1);

    /*
     * Array declaration.
     * ----------------------------------------
     * Name     Type                Description
     * ----------------------------------------
     * psf      Array<precision>    PSFs corresponding to the residuals, see 'lib_array.h' for datatype.
     */

        Array<precision> psf(dims_psf);

    /* ------------------------
     * Loop over MPI processes.
     * ------------------------
     */

        for(int id = 1; id < processes_total; id++){

        /* --------------------------------------------------
         * If rank > number of fried parameters, kill worker.
         * --------------------------------------------------
         */

            if(id > int(dims_residual[0])){

                if(aperture[0] != nullptr){
                
                    MPI_Send(aperture[0], aperture.get_size(), mpi_precision, id, mpi_cmds::kill, MPI_COMM_WORLD);

                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);
                
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                }

            /* -------------------------------------
             * Decrement number of processes in use.
             * -------------------------------------
             */

                processes_total--;
                
            }

        /* -------------------------------------------------------------------
         * If rank < number of fried parameters, send dims_residual to worker.
         * -------------------------------------------------------------------
         */

            else{
                if(aperture[0] != nullptr){

                    MPI_Send(aperture[0], aperture.get_size(), mpi_precision, id, mpi_cmds::task, MPI_COMM_WORLD);

                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);
                
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                
                }
            }

        }

    /* --------------------------------------------------
     * !(4) Distribute residual phase-screens to workers.
     * !(5) Store PSFs returned by workers.
     * !(6) Repeat steps 4-5 for all residual phase-screens.
     * -----------------------------------------------------
     */

        for(int id = 1; id < processes_total; id++){

        /* ----------------------------
         * Send phase-screen residuals.
         * ----------------------------
         */

            if(residual[index_of_fried_in_queue] != nullptr){
                
                MPI_Send(residual[index_of_fried_in_queue], sizeof_vector(dims_residual_per_fried), mpi_precision, id, mpi_cmds::task, MPI_COMM_WORLD);

            }else{

                fprintf(console, "(Error)\tNull buffer, calling MPI_Abort()\n");
                fflush (console);

                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                
            }

        /* -------------------------
         * Update process_fried_map.
         * -------------------------
         */

            process_fried_map[id] = index_of_fried_in_queue;

        /* ------------------------------------
         * Update and display percent_assigned.
         * ------------------------------------
         */

            percent_assigned  = (100.0 * (index_of_fried_in_queue + 1)) / dims_residual[0];
            
            fprintf(console, "\r(Info)\tComputing PSFs:\t\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
            fflush (console);

        /* ----------------------------------
         * Increment index_of_fried_in_queue.
         * ----------------------------------
         */

            index_of_fried_in_queue++;

        }

        while(fried_completed < dims_residual[0]){

        /* --------------------------------------------------------------------
         * Wait for a worker to ping master. If pinged, get worker information.
         * --------------------------------------------------------------------
         */	
	
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, mpi_precision, &mpi_recv_count);
            
        /*
         * Variable declaration:
         * -------------------------------------
         *  Name            Type    Description
         * ------------------------------------
         * fried_index_psf  sizt    Index of next fried in pointer space, for psf array.
         */

        /* -------------------------------------------------
         * Get index of fried parameter processed by worker.
         * -------------------------------------------------
         */

            sizt fried_index_psf = process_fried_map[status.MPI_SOURCE];

        /* ------------------------------------
         * Get PSFs, store at correct location.
         * ------------------------------------
         */

            MPI_Recv(psf[fried_index_psf], sizeof_vector(dims_psf_per_fried), mpi_precision, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        /* --------------------------
         * Increment fried_completed.
         * --------------------------
         */

            fried_completed++;

        /* -------------------------------------
         * Update and display percent_completed.
         * -------------------------------------
         */

            percent_completed  = (100.0 * fried_completed) / dims_residual[0];
           
            fprintf(console, "\r(Info)\tComputing PSFs:\t\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
            fflush (console);

        /* -----------------------------------------------------------------
         * Send next set of residual phase-screens, if available, to worker.
         * -----------------------------------------------------------------
         */

            if(index_of_fried_in_queue < dims_residual[0]){

            /* ----------------------------
             * Send phase-screen residuals.
             * ----------------------------
             */

                if(residual[index_of_fried_in_queue] != nullptr){

                    MPI_Send(residual[index_of_fried_in_queue], sizeof_vector(dims_residual_per_fried), mpi_precision, status.MPI_SOURCE, mpi_cmds::task, MPI_COMM_WORLD);

                }else{

                    fprintf(console, "(Error)\tNull buffer, calling MPI_Abort()\n");
                    fflush (console);

                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                
                }    
	
            /* -------------------------
             * Update process_fried_map.
             * -------------------------
             */
	
                process_fried_map[status.MPI_SOURCE] = index_of_fried_in_queue;

            /* ------------------------------------
             * Update and display percent_assigned.
             * ------------------------------------
             */

                percent_assigned = (100.0 * (index_of_fried_in_queue + 1)) / dims_residual[0];

                fprintf(console, "\r(Info)\tComputing PSFs:\t\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
                fflush (console);

            /* ----------------------------------
             * Increment index_of_fried_in_queue.
             * ----------------------------------
             */

                index_of_fried_in_queue++;
      	    
            }else{
	   
            /* -------------------------------------------------
             * If no more PSFs need to be computed, kill worker.
             * -------------------------------------------------
             */
		        
                if(residual[0] != nullptr){

                    MPI_Send(residual[0], sizeof_vector(dims_residual_per_fried), mpi_precision, status.MPI_SOURCE, mpi_cmds::kill, MPI_COMM_WORLD);
	        
                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);
                
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                
                }

            }

        }

    /* -----------------------
     * !(7) Save PSFs to disk.
     * -----------------------
     */

        fprintf(console, "\n(Info)\tWriting to file:\t[%s, ", config::write_psf_to.c_str());
        fflush (console);

        write_status = psf.wr_fits(config::write_psf_to.c_str(), config::output_clobber);
        if(write_status != EXIT_SUCCESS){
	    
            fprintf(console, "Failed with err code: %d]\n", write_status);
            fflush (console);
	    
        }
	    else{
	    
            fprintf(console, "Done]\n");
            fflush (console);

        }

    }
    
/* -------------------------
 * Workflow for the workers.
 * -------------------------
 */
    
    else if(process_rank){

    /*
     * Vector declaration.
     * ----------------------------------------------------
     * Name                     Type            Description
     * ----------------------------------------------------
     * dims_aperture            sizt_vector     Dimensions of the aperture function.
     * dims_psf_single          sizt_vector     Dimensions of a single PSF.
     * dims_psf_per_fried       sizt_vector     Dimensions of PSFs, per fried.
     * dims_residual_per_fried  sizt_vector     Dimensions of phase-screen residuals, per fried.
     */

        const sizt_vector dims_aperture{config::sims_size_x, config::sims_size_y};
        const sizt_vector dims_psf_single{2 * config::sims_size_x - 1, 2 * config::sims_size_y - 1}; 
        const sizt_vector dims_psf_per_fried{config::sims_per_fried, 2 * config::sims_size_x - 1, 2 * config::sims_size_y - 1};
        const sizt_vector dims_residual_per_fried{config::sims_per_fried, config::sims_size_x, config::sims_size_y};

    /*
     * Array declaration.
     * --------------------------------------------
     * Name         Type                Description
     * --------------------------------------------
     * aperture     Array<precision>    Aperture function.
     */

        Array<precision> aperture{dims_aperture};

    /* ----------------------------------
     * Get aperture function from master.
     * ----------------------------------
     */

        MPI_Recv(aperture[0], aperture.get_size(), mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
    /*
     * Array declaration.
     * --------------------------------------------------------
     * Name                     Type                Description
     * --------------------------------------------------------
     * residual_per_fried       Array<precision>    Phase-screen residuals, per fried.
     * residual_single          Array<precision>    Single residual phase-screen.
     * psf_per_fried            Array<precision>    PSFs of the residuals, per fried.
     * psf_single               Array<precision>    Single PSF.
     * pupil_function           Array<cmpx>         Single pupil function.
     * pupil_function_fourier   Array<cmpx>         Fourier transformed pupil function. 
     */

        Array<precision> residual_per_fried(dims_residual_per_fried);
        Array<precision> residual_single(dims_aperture);
        Array<precision> psf_per_fried(dims_psf_per_fried);
        Array<precision> psf_single(dims_psf_single);
        Array<cmpx>      pupil_function(dims_psf_single);
        Array<cmpx>      pupil_function_fourier(dims_psf_single);

    /* -------------------------------
     * Import fft wisdom if available.
     * -------------------------------
     */

        fftw_import_wisdom_from_filename(config::read_fft_psf_wisdom_from.c_str());

    /*
     * Variable declaration:
     * --------------------------------
     * Name     Type        Description
     * --------------------------------
     * forward  fftw_plan   Re-usable FFTW plan for the forward transformation.
     */

        fftw_plan forward = fftw_plan_dft_2d(dims_psf_single[0], dims_psf_single[1], reinterpret_cast<fftw_complex*>(pupil_function[0]),\
                                             reinterpret_cast<fftw_complex*>(pupil_function_fourier[0]), FFTW_FORWARD, FFTW_MEASURE);

    /* ----------------------------------------------------
     * Compute PSFs of residual phase-screens until killed.
     * ----------------------------------------------------
     */

        while(status.MPI_TAG != mpi_cmds::kill){

        /* ---------------------------------------
         * Get residual phase-screens from master.
         * ---------------------------------------
         */

            MPI_Recv(residual_per_fried[0], residual_per_fried.get_size(), mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if(status.MPI_TAG == mpi_cmds::task){

            /* -------------------------------------------
             * Compute the PSFs of residual phase-screens.
             * -------------------------------------------
             */
                
                for(sizt ind = 0; ind < config::sims_per_fried; ind++){
                    
                /* --------------------------------------------------------------
                 * If airy disk is requested, don't copy into residual_single[0].
                 * --------------------------------------------------------------
                 */
                  
                    if(config::get_airy_disk){
                        
                        make_psf_from_phase_screen(residual_single, psf_single, aperture, forward);                    

                    }
                    else{

                        memcpy(residual_single[0], residual_per_fried[ind], residual_single.get_size() * sizeof(precision));
                        make_psf_from_phase_screen(residual_single, psf_single, aperture, forward); 
                   
                    }                

                /* ---------------------------------------------
                 * Copy psf_single back into psf_per_fried[ind].
                 * ---------------------------------------------
                 */

                    memcpy(psf_per_fried[ind], psf_single[0], psf_single.get_size() * sizeof(precision));

                }
                
                if(psf_per_fried[0] != nullptr){
                
                    MPI_Send(psf_per_fried[0], psf_per_fried.get_size(), mpi_precision, 0, mpi_pmsg::ready, MPI_COMM_WORLD);
                
                }else{

                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                
                }
            }


        /* -------------------------
         * Write FFT wisdom to file.
         * -------------------------
         */
            
            fftw_export_wisdom_to_filename(config::read_fft_psf_wisdom_from.c_str());

        }
    }

    MPI_Finalize();
    return(EXIT_SUCCESS);
}
