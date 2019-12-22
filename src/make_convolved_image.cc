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
 * This program convolves an input image with the Point Spread Function (PSF) corresponding to phase-screen
 * residuals. The convolution is implemented as a multiplication in fourier space.
 *
 * ------
 * Usage:
 * ------
 * mpiexec -np <cores> ./make_convolve_image <config_file>
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
 * 2. Read the PSFs from file.
 * 3. Read the input image from file.
 * 4. Distribute PSFs to workers.
 * 5. Store the convolved images returned by workers.
 * 6. Repeat steps 4-5 for all PSFs.
 * 7. Save convolved images to disk.
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

/* ---------------
 * Initialize MPI.
 * ---------------
 */
   
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processes_total);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

/* -----------------------------------------------------------
 * Only the master MPI process - rank zero - prints to stdout.
 * -----------------------------------------------------------
 */

    FILE *console   = process_rank == 0 ? stdout : fopen("/dev/null","wb");
    fprintf(console, "-----------------------------\n");
    fprintf(console, "- Image convolution program -\n");
    fprintf(console, "-----------------------------\n");

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
     * Name     Type                Description
     * --------------------------------------------
     * psf      Array<precision>    PSFs of residual phase-screens, see 'lib_array.h' for datatype.
     * image    Array<precision>    Image to be convolved, see 'lib_array.h' for datatype.
     */

        Array<precision> psf;
        Array<precision> image;

    /* -----------------------------
     * !(2) Read the PSFs from file.
     * -----------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::write_psf_to.c_str());
        fflush (console);

        read_status = psf.rd_fits(config::write_psf_to.c_str());
        if(read_status != EXIT_SUCCESS){
            
            fprintf(console, "Failed with err code: %d]\n", read_status);
            fflush (console);

            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        
        }else{
            
            fprintf(console, "Done]\n");
            fflush (console);

        }

    /* ------------------------------
     * !(3) Read the image from file.
     * ------------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::read_image_from.c_str());
        fflush (console);

        read_status = image.rd_fits(config::read_image_from.c_str());
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
     * dims_psf                 sizt_vector     Dimensions of PSFs.
     * dims_image               sizt_vector     Dimensions of PSFs.
     * dims_psf_per_fried       sizt_vector     Dimensions of PSFs, per_fried.
     * dims_convolved_images    sizt_vector     Dimensions of the convolved images array.
     * dims_images_per_fried    sizt_vector     Dimensions of the convolved images array, per fried.
     * process_fried_map        sizt_vector     Map of which process is handling which fried index.
     */

        const sizt_vector dims_psf   = psf.get_dims();
        const sizt_vector dims_image = image.get_dims();
        const sizt_vector dims_psf_per_fried(dims_psf.begin() + 1, dims_psf.end());
        const sizt_vector dims_convolved_images{dims_psf[0], dims_psf[1], dims_image[0], dims_image[1]};
        const sizt_vector dims_images_per_fried{dims_psf[1], dims_image[0], dims_image[1]};
        sizt_vector process_fried_map(dims_psf[0] + 1);

    /*
     * Array declaration.
     * ----------------------------------------------------
     * Name                 Type                Description
     * ----------------------------------------------------
     * convolved_images     Array<precision>    Convolved images, see 'lib_array.h' for datatype.
     */

        Array<precision> convolved_images(dims_convolved_images);

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
                
                    MPI_Send(dims_psf_per_fried.data(), dims_psf_per_fried.size(), mpi_precision, id, mpi_cmds::kill, MPI_COMM_WORLD);
                    MPI_Send(dims_image.data(), dims_image.size(), mpi_precision, id, mpi_cmds::kill, MPI_COMM_WORLD);

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

                    MPI_Send(dims_psf_per_fried.data(), dims_psf_per_fried.size(), mpi_precision, id, mpi_cmds::task, MPI_COMM_WORLD);
                    MPI_Send(dims_image.data(), dims_image.get_size(), mpi_precision, id, mpi_cmds::task, MPI_COMM_WORLD);

                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);
                
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                
                }
            }

        }

    /* ----------------------------
     * Distribute image to workers.
     * ----------------------------
     */

        for(int id = 1; id < processes_total; id++){

            if(image[0] != nullptr){

                MPI_Send(image[0], image.get_size(), mpi_precision,  id, mpi_cmds::task, MPI_COMM_WORLD);

            }else{

                fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                fflush (console);

                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

    /* ------------------------------------
     * !(4) Distribute the PSFs to workers.
     * !(5) Store convolved images returned  by workers.
     * !(6) Repeat steps 4-5 for all PSFs.
     * -----------------------------------
     */

        for(int id = 1; id < processes_total; id++){

        /* -------------------------
         * Send the PSFs to workers.
         * -------------------------
         */

            if(psf[index_of_fried_in_queue] != nullptr){
                
                MPI_Send(psf[index_of_fried_in_queue], sizeof_vector(dims_psf_per_fried), mpi_precision, id, mpi_cmds::task, MPI_COMM_WORLD);

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

            percent_assigned  = (100.0 * (index_of_fried_in_queue + 1)) / dims_psf[0];
            
            fprintf(console, "\r(Info)\tComputing PSFs:\t\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
            fflush (console);

        /* ----------------------------------
         * Increment index_of_fried_in_queue.
         * ----------------------------------
         */

            index_of_fried_in_queue++;

        }

        while(fried_completed < dims_psf[0]){

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
         * fried_index_image  sizt    Index of next fried in pointer space, for image array.
         */

        /* -------------------------------------------------
         * Get index of fried parameter processed by worker.
         * -------------------------------------------------
         */

            sizt fried_index_image = process_fried_map[status.MPI_SOURCE];

        /* ------------------------------------
         * Get PSFs, store at correct location.
         * ------------------------------------
         */

            MPI_Recv(convolved_images[fried_index_psf], sizeof_vector(dims_images_per_fried), mpi_precision, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        /* --------------------------
         * Increment fried_completed.
         * --------------------------
         */

            fried_completed++;

        /* -------------------------------------
         * Update and display percent_completed.
         * -------------------------------------
         */

            percent_completed  = (100.0 * fried_completed) / dims_psf[0];
           
            fprintf(console, "\r(Info)\tComputing PSFs:\t\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
            fflush (console);

        /* -----------------------------------------------
         * Send next set of PSFs, if available, to worker.
         * -----------------------------------------------
         */

            if(index_of_fried_in_queue < dims_psf[0]){

            /* -------------------------
             * Send the PSFs to workers.
             * -------------------------
             */

                if(psf[index_of_fried_in_queue] != nullptr){

                    MPI_Send(psf[index_of_fried_in_queue], sizeof_vector(dims_psf_per_fried), mpi_precision, status.MPI_SOURCE, mpi_cmds::task, MPI_COMM_WORLD);

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

                percent_assigned = (100.0 * (index_of_fried_in_queue + 1)) / dims_psf[0];

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
		        
                if(psf[0] != nullptr){

                    MPI_Send(psf[0], sizeof_vector(dims_psf_per_fried), mpi_precision, status.MPI_SOURCE, mpi_cmds::kill, MPI_COMM_WORLD);
	        
                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);
                
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                
                }

            }

        }

    /* -----------------------------------
     * !(7) Save convolved images to disk.
     * -----------------------------------
     */

        fprintf(console, "\n(Info)\tWriting to file:\t[%s, ", config::write_images_to.c_str());
        fflush (console);

        write_status = convolved_images.wr_fits(config::write_images_to.c_str(), config::output_clobber);
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
     * dims_image_single        sizt_vector     Dimensions of a single image.
     * dims_psf_per_fried       sizt_vector     Dimensions of PSFs, per fried.
     */

        const sizt_vector dims_psf_per_fried{1, 1, 1}; 
        const sizt_vector dims_image_single{1, 1};
    
    /* --------------------------------------------
     * Get PSFs dimensions, per fried, from master.
     * --------------------------------------------
     */

        MPI_Recv(dims_psf_per_fried.data(), dims_psf_per_fried.size(), mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    
    /* ---------------------------------
     * Get image dimensions from master.
     * ---------------------------------
     */

        MPI_Recv(dims_image_single.data(), dims_image_single.size(), mpi_precision, 0, mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
    /*
     * Vector declaration.
     * ----------------------------------------------------
     * Name                     Type            Description
     * ----------------------------------------------------
     * dims_images_per_fried    sizt_vector     Dimensions of convolved images, per fried.
     * dims_psf_single          sizt_vector     Dimensions of a single PSF.
     */
        const sizt_vector dims_images_per_fried{dims_psf_per_fried[1], dims_image_single[0], dims_image_single[1]};
        const sizt_vector dims_psf_single{dims_psf_per_fried.begin() + 1, dims_psf_per_fried.end()};
 
    /*
     * Array declaration.
     * --------------------------------------------------------
     * Name                     Type                Description
     * --------------------------------------------------------
     * psf_per_fried            Array<precision>    PSFs of the residuals, per fried.
     * psf_single               Array<precision>    Single PSF.
     * image_single             Array<precision>    Image to be convolved.
     * images_per_fried         Array<precision>    Convolved images, per fried.
     * image_single_fourier     Array<cmpx>         Fourier of the image.
     * psf_single_fourier       Array<cmpx>         Fourier of the PSF. 
     */

        Array<precision> psf_per_fried(dims_psf_per_fried);
        Array<precision> psf_single(dims_psf_single);
        Array<precision> image_single(dims_image_single);
        Array<precision> images_per_fried(dims_images_per_fried);
        Array<cmpx>      image_single_fourier{dims_image_single};
        Array<cmpx>      psf_single_fourier{dims_psf_single};      

    /* ----------------------
     * Get image from master.
     * ----------------------
     */

        MPI_Recv(image_single[0], image_single.get_size(), mpi_precision, 0, mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        

    /* -------------------------------
     * Import fft wisdom if available.
     * -------------------------------
     */

        fftw_import_wisdom_from_filename(config::read_fft_image_wisdom_from.c_str());

    /*
     * Variable declaration:
     * --------------------------------
     * Name     Type        Description
     * --------------------------------
     * forward  fftw_plan   Re-usable FFTW plan for the forward transformation.
     */

        fftw_plan psf_forward   = fftw_plan_dft_2d_r2c(dims_psf_single[0], dims_psf_single[1], psf_single_fourier[0],\
                                                       reinterpret_cast<fftw_complex*>(psf_single_fourier[0]), FFTW_FORWARD, FFTW_MEASURE);

        fftw_plan image_forward = fftw_plan_dft_2d_r2c(dims_image_single[0], dims_image_single[1], image_single_fourier[0]),\
                                                   reinterpret_cast<fftw_complex*>(image_single_fourier[0]), FFTW_FORWARD, FFTW_MEASURE);

        fftw_plan image_reverse = fftw_plan_dft_2d_c2r(dims_image_single[0], dims_image_single[1], reinterpret_cast<fftw_complex*>(image_single_fourier[0]), 
                                                       convolved_image_single[0], FFTW_BACKWARD, FFTW_MEASURE);

    /* ---------------------------------------
     * Convolve image with PSFs, until killed.
     * ---------------------------------------
     */

        while(status.MPI_TAG != mpi_cmds::kill){

        /* ---------------------------------------
         * Get residual phase-screens from master.
         * ---------------------------------------
         */

            MPI_Recv(psf_per_fried[0], psf_per_fried.get_size(), mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if(status.MPI_TAG == mpi_cmds::task){

            /* -----------------------------
             * Compute the convolved images.
             * -----------------------------
             */
                
                for(sizt ind = 0; ind < config::sims_per_fried; ind++){
                    

                }
                
                if(images_per_fried[0] != nullptr){
                
                    MPI_Send(images_per_fried[0], images_per_fried.get_size(), mpi_precision, 0, mpi_pmsg::ready, MPI_COMM_WORLD);
                
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
