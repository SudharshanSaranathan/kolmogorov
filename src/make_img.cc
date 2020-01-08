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
 * This program convolves an input image with the Point Spread Functions (PSFs) corresponding to residual phase-screens.
 * The convolution is implemented as a multiplication in fourier space.
 *
 * ------
 * Usage:
 * ------
 * mpiexec -np <cores> ./make_img <config_file>
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
 *  mpi_precision   MPI_Datatype   MPI_FLOAT or MPI_DOUBLE.
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
 * ------------------------------------
 * Workflow for MPI process with rank 0
 * ------------------------------------
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
     * img      Array<precision>    Image to be convolved, see 'lib_array.h' for datatype.
     * psfs     Array<precision>    PSFs of residual phase-screens, see 'lib_array.h' for datatype.
     */

        Array<precision> img;
        Array<precision> psfs;

    /* -----------------------------
     * !(2) Read the PSFs from file.
     * -----------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::write_psf_to.c_str());
        fflush (console);

        read_status = psfs.rd_fits(config::write_psf_to.c_str());
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

        read_status = img.rd_fits(config::read_image_from.c_str());
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
     * dims_psfs                sizt_vector     Dimensions of PSFs.
     * dims_psfs_per_fried      sizt_vector     Dimensions of PSFs, per_fried.
     * dims_img                 sizt_vector     Dimensions of the image.
     * dims_imgs                sizt_vector     Dimensions of the convolved images.
     * dims_imgs_per_fried      sizt_vector     Dimensions of the convolved images, per fried.
     * process_fried_map        sizt_vector     Map of which process is handling which fried index.
     */

        const sizt_vector dims_psfs = psfs.get_dims();
        const sizt_vector dims_psfs_per_fried(dims_psfs.begin() + 1, dims_psfs.end());
        const sizt_vector dims_img = img.get_dims();
        sizt_vector dims_imgs(dims_psfs);
        
        dims_imgs.rbegin()[0] = dims_img.rbegin()[0];
        dims_imgs.rbegin()[1] = dims_img.rbegin()[1];

        const sizt_vector dims_imgs_per_fried(dims_imgs.begin() + 1, dims_imgs.end());
        sizt_vector       process_fried_map(dims_psfs[0] + 1);

    /* --------------------------
     * Validate image dimensions.
     * --------------------------
     */

        if(dims_img.size() != 2){

            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

        }

    /*
     * Array declaration.
     * ----------------------------------------
     * Name     Type                Description
     * ----------------------------------------
     * imgs     Array<precision>    Convolved images, see 'lib_array.h' for datatype.
     */

        Array<precision> imgs(dims_imgs);

    /*
     * Variable declaration.
     * ------------------------------------------------
     * Name                         Type    Description
     * ------------------------------------------------
     * dims_psfs_per_fried_naxis     sizt    Number of dimensions of psf, per fried.
     */

        sizt dims_psfs_per_fried_naxis = dims_psfs_per_fried.size();

    /* ------------------------
     * Loop over MPI processes.
     * ------------------------
     */

        for(int id = 1; id < processes_total; id++){

        /* --------------------------------------------------
         * If rank > number of fried parameters, kill worker.
         * --------------------------------------------------
         */

            if(id > int(dims_psfs[0])){

                MPI_Send(&dims_psfs_per_fried_naxis,  1, MPI_UNSIGNED_LONG, id, mpi_cmds::kill, MPI_COMM_WORLD);
                MPI_Send( dims_psfs_per_fried.data(), dims_psfs_per_fried.size(), MPI_UNSIGNED_LONG, id, mpi_cmds::kill, MPI_COMM_WORLD);
                MPI_Send( dims_img.data(), dims_img.size(), MPI_UNSIGNED_LONG, id, mpi_cmds::kill, MPI_COMM_WORLD);

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
            
                MPI_Send(&dims_psfs_per_fried_naxis,  1, MPI_UNSIGNED_LONG, id, mpi_cmds::task, MPI_COMM_WORLD);
                MPI_Send( dims_psfs_per_fried.data(), dims_psfs_per_fried.size(), MPI_UNSIGNED_LONG, id, mpi_cmds::task, MPI_COMM_WORLD);
                MPI_Send( dims_img.data(), dims_img.size(), MPI_UNSIGNED_LONG, id, mpi_cmds::task, MPI_COMM_WORLD);
            
            }

        }

    /* ----------------------------
     * Distribute image to workers.
     * ----------------------------
     */

        for(int id = 1; id < processes_total; id++){

            if(img[0] != nullptr){
                
                MPI_Send(img[0], img.get_size(), mpi_precision,  id, mpi_cmds::task, MPI_COMM_WORLD);

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

            if(psfs[index_of_fried_in_queue] != nullptr){
                
                MPI_Send(psfs[index_of_fried_in_queue], sizeof_vector(dims_psfs_per_fried), mpi_precision, id, mpi_cmds::task, MPI_COMM_WORLD);

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

            percent_assigned  = (100.0 * (index_of_fried_in_queue + 1)) / dims_psfs[0];
            
            fprintf(console, "\r(Info)\tConvolving image:\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
            fflush (console);

        /* ----------------------------------
         * Increment index_of_fried_in_queue.
         * ----------------------------------
         */

            index_of_fried_in_queue++;

        }

        while(fried_completed < dims_psfs[0]){

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

            MPI_Recv(imgs[fried_index_image], sizeof_vector(dims_imgs_per_fried), mpi_precision, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        /* --------------------------
         * Increment fried_completed.
         * --------------------------
         */

            fried_completed++;

        /* -------------------------------------
         * Update and display percent_completed.
         * -------------------------------------
         */

            percent_completed  = (100.0 * fried_completed) / dims_psfs[0];
           
            fprintf(console, "\r(Info)\tConvolving image:\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
            fflush (console);

        /* -----------------------------------------------
         * Send next set of PSFs, if available, to worker.
         * -----------------------------------------------
         */

            if(index_of_fried_in_queue < dims_psfs[0]){

            /* -------------------------
             * Send the PSFs to workers.
             * -------------------------
             */

                if(psfs[index_of_fried_in_queue] != nullptr){

                    MPI_Send(psfs[index_of_fried_in_queue], sizeof_vector(dims_psfs_per_fried), mpi_precision, status.MPI_SOURCE, mpi_cmds::task, MPI_COMM_WORLD);

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

                percent_assigned = (100.0 * (index_of_fried_in_queue + 1)) / dims_psfs[0];

                fprintf(console, "\r(Info)\tConvolving image:\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
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
		        
                if(psfs[0] != nullptr){
                    MPI_Send(psfs[0], sizeof_vector(dims_psfs_per_fried), mpi_precision, status.MPI_SOURCE, mpi_cmds::kill, MPI_COMM_WORLD);
	        
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

        write_status = imgs.wr_fits(config::write_images_to.c_str(), config::output_clobber);
        if(write_status != EXIT_SUCCESS){
	    
            fprintf(console, "Failed with err code: %d]\n", write_status);
            fflush (console);
	    
        }
	    else{
	    
            fprintf(console, "Done]\n");
            fflush (console);

        }

    }
    
/* -----------------------------------------
 * Workflow for MPI processes with rank > 0.
 * -----------------------------------------
 */
    
    else if(process_rank){

    /*
     * Variable declaration.
     * ------------------------------------
     * Name             Type    Description
     * ------------------------------------
     * dims_psfs_naxis  sizt    Dimensionality of the PSFs array.
     */

        sizt dims_psfs_naxis;

    /*
     * Vector declaration.
     * ----------------------------------------------------
     * Name         Type            Description
     * ----------------------------------------------------
     * dims_psf     sizt_vector     Dimensions of a single, 2D PSF. 
     * dims_img     sizt_vector     Dimensions of the 2D image to be convolved.
     * dims_psfs    sizt_vector     Dimensions of the PSFs, per fried.
     */

        sizt_vector dims_psf(2);
        sizt_vector dims_img(2);
        sizt_vector dims_psfs;

    /* -------------------------------------------------------
     * Get dimensionality of the PSFs array from master.
     * -------------------------------------------------------
     */

        MPI_Recv(&dims_psfs_naxis, 1, MPI_UNSIGNED_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    /* ---------------------------
     * Resize dims_psfs_per_fried.
     * ---------------------------
     */
        dims_psfs.resize(dims_psfs_naxis);
    
    /* ---------------------------------------------------
     * Get dimensions of the psfs, per fried, from master.
     * ---------------------------------------------------
     */

        MPI_Recv(dims_psfs.data(), dims_psfs.size(), MPI_UNSIGNED_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    
    /* ----------------------------------------
     * Get dimensions of the image from master.
     * ----------------------------------------
     */

        MPI_Recv(dims_img.data(), dims_img.size(), MPI_UNSIGNED_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
    /*
     * Vector declaration.
     * ----------------------------------------
     * Name         Type            Description
     * ----------------------------------------
     * dims_imgs    sizt_vector     Dimensions of the convolved images.
     * dims_shift   sizt_vector     Fourier transform shifts.
     */

        sizt_vector dims_imgs(dims_psfs);
        sizt_vector dims_shift(dims_img);
 
        dims_imgs.rbegin()[0] = dims_img.rbegin()[0];
        dims_imgs.rbegin()[1] = dims_img.rbegin()[1];

        dims_psf.rbegin()[0] = dims_psfs.rbegin()[0];
        dims_psf.rbegin()[1] = dims_psfs.rbegin()[1];

        dims_shift[0] /= 2;
        dims_shift[1] /= 2;

    /*
     * Variable declaration.
     * ----------------------------------------
     * Name             Type    Description
     * ----------------------------------------
     * img_center_x     sizt    Center of the image.
     * img_center_y     sizt    Center of the image.
     * psf_center_x     sizt    Center of the PSF.
     * psf_center_y     sizt    Center of the PSF.
     */

        sizt img_center_x = dims_img[0] / 2;
        sizt img_center_y = dims_img[1] / 2;
        sizt psf_center_x = dims_psf[0] / 2;
        sizt psf_center_y = dims_psf[1] / 2;

    /*
     * Array declaration.
     * --------------------------------------------------------
     * Name             Type                Description
     * --------------------------------------------------------
     * img              Array<precision>    Image to be convolved.
     * imgs             Array<precision>    Image convolved for each PSF.
     * psfs             Array<precision>    Point spread functions corresponding to the residual phase-screens, per fried.
     * psf_fourier      Array<cmpx>         Fourier transform of the PSF.
     * img_fourier      Array<cmpx>         Fourier transform of the image.
     * img_fourier_c    Array<cmpx>         Fourier transform of the convolved image.
     */

        Array<precision> img(dims_img);
        Array<precision> imgs(dims_imgs);
        Array<precision> psfs(dims_psfs);
        Array<cmpx>      psf_fourier(dims_img);
        Array<cmpx>      img_fourier(dims_img);
        Array<cmpx>      img_fourier_c(dims_img);

    /* --------------------------------------------------------------
     * If the MPI process has not been killed, get image from master.
     * --------------------------------------------------------------
     */

        if(status.MPI_TAG != mpi_cmds::kill){

            MPI_Recv(img[0], img.get_size(), mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        } 

    /*
     * Array declaration.
     * ----------------------------------------
     * Name             Type            Description
     * ----------------------------------------
     * img_double       Array<double>   Double precision image.
     * psf_double       Array<double>   Double precision PSF.
     * img_double_c     Array<double>   Double precision convolved image.
     */

        Array<double> img_double(dims_img);
        Array<double> psf_double(dims_img);
        Array<double> img_double_c(dims_img);

    /* -------------------------------
     * Copy image to double precision.
     * -------------------------------
     */

        img.cast_to_type(img_double);

    /*
     * Variable declaration:
     * --------------------------------
     * Name         Type        Description
     * --------------------------------
     * psf_forward  fftw_plan   Re-usable FFTW plan for the forward transformation of psf.
     * img_forward  fftw_plan   Re-usable FFTW plan for the forward transformation of img.
     * img_reverse  fftw_plan   Re-usable FFTW plan for the reverse transformation of img.
     */
    
        fftw_plan psf_forward = fftw_plan_dft_r2c_2d(dims_img[0], dims_img[1], psf_double[0], reinterpret_cast<fftw_complex*>(psf_fourier[0]), FFTW_MEASURE);
        fftw_plan img_forward = fftw_plan_dft_r2c_2d(dims_img[0], dims_img[1], img_double[0], reinterpret_cast<fftw_complex*>(img_fourier[0]), FFTW_MEASURE);
        fftw_plan img_reverse = fftw_plan_dft_c2r_2d(dims_img[0], dims_img[1], reinterpret_cast<fftw_complex*>(img_fourier_c[0]), img_double_c[0], FFTW_MEASURE);

    /* ----------------------------------------
     * Compute the fourier of the double image.
     * ----------------------------------------
     */

        fftw_execute(img_forward);

    /* -------------------------------------------
     * Convolve the image with PSFs, until killed.
     * -------------------------------------------
     */

        while(status.MPI_TAG != mpi_cmds::kill){

        /* --------------------------------
         * Get the PSFs from MPI rank zero.
         * --------------------------------
         */

            MPI_Recv(psfs[0], psfs.get_size(), mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if(status.MPI_TAG == mpi_cmds::task){

                if(dims_psfs.size() == 2){
                    
                    for(sizt xpix = 0; xpix < dims_psf[0]; xpix++){
                        for(sizt ypix = 0; ypix < dims_psf[1]; ypix++){
                            psf_double(xpix + img_center_x - psf_center_x, ypix + img_center_y - psf_center_y) = static_cast<double>(psfs(xpix, ypix));
                        }
                    }

                    fftw_execute_dft_r2c(psf_forward, psf_double[0], reinterpret_cast<fftw_complex*>(psf_fourier[0]));
                    img_fourier_c = img_fourier * psf_fourier;
                    fftw_execute_dft_c2r(img_reverse, reinterpret_cast<fftw_complex*>(img_fourier_c[0]), img_double_c[0]);

                    img_double_c = img_double_c.roll(dims_shift) / img_double_c.get_size();                   
                    img_double_c.cast_to_type(imgs);   

                }else if(dims_psfs.size() == 3){
                    for(sizt ind = 0; ind < dims_psfs[0]; ind++){

                        for(sizt xpix = 0; xpix < dims_psf[0]; xpix++){
                            for(sizt ypix = 0; ypix < dims_psf[1]; ypix++){
                                psf_double(xpix + img_center_x - psf_center_x, ypix + img_center_y - psf_center_y) = psfs(ind, xpix, ypix);
                            }
                        }
                    
                        fftw_execute(psf_forward);
                        img_fourier_c = img_fourier * psf_fourier;
                        fftw_execute_dft_c2r(img_reverse, reinterpret_cast<fftw_complex*>(img_fourier_c[0]), img_double_c[0]);

                        img_double_c = img_double_c.roll(dims_shift) / img_double_c.get_size();                   
                        for(sizt xpix = 0; xpix < dims_img[0]; xpix++){
                            for(sizt ypix = 0; ypix < dims_img[1]; ypix++){
                                imgs(ind, xpix, ypix) = static_cast<precision>(img_double_c(xpix, ypix));
                            }
                        }
                    }
                }

                MPI_Send(imgs[0], imgs.get_size(), mpi_precision, 0, mpi_pmsg::ready, MPI_COMM_WORLD);

            }
        }
    }

    MPI_Finalize();
    return(EXIT_SUCCESS);
}
