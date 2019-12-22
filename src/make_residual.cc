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
 * This program subtracts basis functions defined on the aperture from phase-screen simulations.
 * The subtraction is weighted, if weights are provided. The residual phase-screens represent
 * corrections to the phase-screens either by Adaptive Optics (AO) or by post-facto image
 * processing. 
 *
 * ------
 * Usage:
 * ------
 * mpiexec -np <cores> ./make_residual <config_file>
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
 * 2. Read phase-screen simulations from file.
 * 3. Read basis functions from file.
 * 4. Read basis weights from file.
 * 5. Distribute phase-screen simulations to workers.
 * 6. Store residual phase-screens returned by workers.
 * 7. Repeat steps 5-6 for all phase-screen simulations.
 * 8. Save residual phase-screens to disk.
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
 *  Name            Type            Description
 *  -------------------------------------------
 *  status          MPI_status    See MPI documentation.
 *  mpi_precision   MPI::Datatype   MPI::FLOAT or MPI::DOUBLE.
 *  process_rank    int             Rank of MPI processes.
 *  process_total   int             Store the total number of MPI processes
 *  mpi_recv_count  int             Store the count of data received in MPI_Recv, see MPI documentation for explanation.
 *  read_status     int             File read status.
 *  write_status    int             File write status.
 */
   
    MPI_Status status;
    MPI::Datatype mpi_precision = std::is_same<precision, float>::value == true ? MPI::FLOAT : MPI::DOUBLE;
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
    fprintf(console, "------------------------------------------------------\n");
    fprintf(console, "- Phase-screen residuals computation program -\n");
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
     * ----------------------------------------
     * Name     Type                Description
     * ----------------------------------------
     * phase    Array<precision>    Phase-screen simulations, see 'lib_array.h' for datatype.
     * basis    Array<precision>    Basis functions on aperture.
     * weights  Array<precision>    Basis weights.
     */

        Array<precision> phase;
        Array<precision> basis;
        Array<precision> weights;

    /* ---------------------------------------------
     * !(2) Read phase-screen simulations from file.
     * ---------------------------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::write_phase_to.c_str());
        fflush (console);

        read_status = phase.rd_fits(config::write_phase_to.c_str());
        if(read_status != EXIT_SUCCESS){
            
            fprintf(console, "Failed with err code: %d]\n", read_status);
            fflush (console);
            
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

        }else{
            
            fprintf(console, "Done]\n");
            fflush (console);

        }

    /* ------------------------------------
     * !(3) Read basis functions from file.
     * ------------------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::read_basis_from.c_str());
        fflush (console);

        read_status = basis.rd_fits(config::read_basis_from.c_str());
        if(read_status != EXIT_SUCCESS){
        
            fprintf(console, "Failed with err code: %d]\n", read_status);
            fflush (console);
        
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

        }else{
            
            fprintf(console, "Done]\n");
            fflush (console);

        }

    /* ----------------------------------
     * !(4) Read basis weights from file.
     * ----------------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::read_weights_from.c_str());
        fflush (console);
        
        read_status = weights.rd_fits(config::read_weights_from.c_str());
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
     * ------------------------------------------------
     * Name                 Type            Description
     * ------------------------------------------------
     * dims_phase           sizt_vector     Dimensions of phase-screen simulations.
     * dims_basis           sizt_vector     Dimensions of basis functions.
     * dims_weights         sizt_vector     Dimensions of basis weights.
     * dims_phase_per_fried sizt_vector     Dimensions of phase-screen simulations, per fried.
     * process_fried_map    sizt_vector     Map of which process is handling which fried index.
     */

        const sizt_vector dims_phase   = phase.get_dims();
        const sizt_vector dims_basis   = basis.get_dims();
        const sizt_vector dims_weights = weights.get_dims();
        const sizt_vector dims_phase_per_fried(dims_phase.begin() + 1, dims_phase.end());
        sizt_vector process_fried_map(dims_phase[0] + 1);

    /* --------------------------------------------
     * Validate the dimensions of the input arrays.
     * --------------------------------------------
     */

        if(dims_phase[2] != config::sims_size_x  || dims_phase[3] != config::sims_size_y){
     
            fprintf(console, "(Error)\texpected phase-screens with size [%ud %ud], calling MPI_Abort()\n", config::sims_size_x, config::sims_size_y);
            fflush (console);

            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

        }

        if(dims_basis[1] != config::sims_size_x  || dims_basis[2] != config::sims_size_y){
     
            fprintf(console, "(Error)\texpected basis functions with size [%ud %ud], calling MPI_Abort()\n", config::sims_size_x, config::sims_size_y);
            fflush (console);
            
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

        }

        if(dims_weights[0] != dims_phase[0] || dims_weights[1] != dims_basis[0]){
            
            fprintf(console, "(Error)\texpected weights with dimensions [%lu %lu], calling MPI_Abort()\n", dims_phase[0], dims_basis[0]);
            fflush (console);
            
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    
        }

     /*
     * Array declaration.
     * --------------------------------------------
     * Name         Type                Description
     * --------------------------------------------
     * residual     Array<precision>    Residual phase-screens, see 'lib_array.h' for datatype.
     */

        Array<precision> residual(phase);

    /*
     * Variable declaration.
     * ----------------------------------------
     * Name                 Type    Description
     * ----------------------------------------
     * dims_basis_naxis     sizt    Number of axes of dims_basis.
     */

        sizt dims_basis_naxis  = dims_basis.size();

    /* ------------------------
     * Loop over MPI processes.
     * ------------------------
     */

        for(int id = 1; id < processes_total; id++){

        /* --------------------------------------------------
         * If rank > number of fried parameters, kill worker.
         * --------------------------------------------------
         */

            if(id > int(dims_phase[0])){

                MPI_Send(&dims_basis_naxis, 1, MPI::UNSIGNED_LONG, id, mpi_cmds::kill, MPI_COMM_WORLD);
                MPI_Send( dims_basis.data(), dims_basis_naxis, MPI::UNSIGNED_LONG, id, mpi_cmds::kill, MPI_COMM_WORLD);

            /* -------------------------------------
             * Decrement number of processes in use.
             * -------------------------------------
             */

                processes_total--;
                
            }

        /* ----------------------------------------------------------------
         * If rank < number of fried parameters, send dims_basis to worker.
         * ----------------------------------------------------------------
         */

            else{

                MPI_Send(&dims_basis_naxis, 1, MPI::UNSIGNED_LONG, id, mpi_cmds::task, MPI_COMM_WORLD);
                MPI_Send( dims_basis.data(), dims_basis_naxis, MPI::UNSIGNED_LONG, id, mpi_cmds::task, MPI_COMM_WORLD);

            }

        }

    /* --------------------------------
     * Send basis functions to workers.
     * --------------------------------
     */

        for(int id = 1; id < processes_total; id++){

            if(basis[0] != nullptr){

                MPI_Send(basis[0], basis.get_size(), mpi_precision, id, mpi_cmds::task, MPI_COMM_WORLD);

            }else{

                fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                fflush (console);
                
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

            }
        }

    /* ----------------------------------------------------
     * !(5) Distribute phase-screen simulations to workers.
     * !(6) Store residual phase-screens returned by workers.
     * !(7) Repeat steps 5-6 for all phase-screen simulations.
     * -------------------------------------------------------
     */

        for(int id = 1; id < processes_total; id++){

        /* -------------------
         * Send basis weights.
         * -------------------
         */

            if(weights[index_of_fried_in_queue] != nullptr){
                
                MPI_Send(weights[index_of_fried_in_queue], dims_basis[0], mpi_precision, id, mpi_cmds::task, MPI_COMM_WORLD);

            }else{

                fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                fflush (console);
                
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

            }

        /* ------------------------------
         * Send phase-screen simulations.
         * ------------------------------
         */

            if(phase[index_of_fried_in_queue] != nullptr){
                
                MPI_Send(phase[index_of_fried_in_queue], sizeof_vector(dims_phase_per_fried), mpi_precision, id, mpi_cmds::task, MPI_COMM_WORLD);

            }else{

                fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
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

            percent_assigned  = (100.0 * (index_of_fried_in_queue + 1)) / dims_phase[0];
            
            fprintf(console, "\r(Info)\tComputing residuals: \t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
            fflush (console);

        /* ----------------------------------
         * Increment index_of_fried_in_queue.
         * ----------------------------------
         */

            index_of_fried_in_queue++;

        }

        while(fried_completed < dims_phase[0]){

        /* --------------------------------------------------------------------
         * Wait for a worker to ping master. If pinged, get worker information.
         * --------------------------------------------------------------------
         */	
	
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, mpi_precision, &mpi_recv_count);

        /*
         * Variable declaration:
         *-----------------------------------------
         * Name                 Type    Description
         * ----------------------------------------
         * fried_index_weights  sizt    Index of next fried in pointer space, for array weights.
         * fried_index_phase    sizt    Index of next fried in pointer space, for array phase.
         */

        /* -------------------------------------------------
         * Get index of fried parameter processed by worker.
         * -------------------------------------------------
         */	
	        
            sizt fried_index_phase   = process_fried_map[status.MPI_SOURCE];

        /* ------------------------------------------------------
         * Get residual phase-screens, store at correct location.
         * ------------------------------------------------------
         */

            MPI_Recv(residual[fried_index_phase], sizeof_vector(dims_phase_per_fried), mpi_precision, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        /* --------------------------
         * Increment fried_completed.
         * --------------------------
         */

            fried_completed++;

        /* -------------------------------------
         * Update and display percent_completed.
         * -------------------------------------
         */

            percent_completed  = (100.0 * fried_completed) / dims_phase[0];
            
            fprintf(console, "\r(Info)\tComputing residuals:\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
            fflush (console);

        /* -------------------------------------------------------------------
         * Send next set of phase-screen simulations, if available, to worker.
         * -------------------------------------------------------------------
         */

            if(index_of_fried_in_queue < dims_phase[0]){


           /* -------------------
            * Send basis weights.
            * -------------------
            */

                if(weights[index_of_fried_in_queue] != nullptr){
                
                    MPI_Send(weights[index_of_fried_in_queue], dims_basis[0], mpi_precision, status.MPI_SOURCE, mpi_cmds::task, MPI_COMM_WORLD);

                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);
            
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                }

           /* -------------------
            * Send phase-screens.
            * -------------------
            */

                if(phase[index_of_fried_in_queue] != nullptr){
                
                    MPI_Send(phase[index_of_fried_in_queue], sizeof_vector(dims_phase_per_fried), mpi_precision, status.MPI_SOURCE, mpi_cmds::task, MPI_COMM_WORLD);

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

            /* ------------------------------------
             * Update and display percent_assigned.
             * ------------------------------------
             */

                percent_assigned = (100.0 * (index_of_fried_in_queue + 1)) / dims_phase[0];
                
                fprintf(console, "\r(Info)\tComputing residuals:\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
                fflush (console);

            /* ----------------------------------
             * Increment index_of_fried_in_queue.
             * ----------------------------------
             */

                index_of_fried_in_queue++;
      	    
            }else{
	    
            /* ------------------------------------------------------
             * If no more residuals need to be computed, kill worker.
             * ------------------------------------------------------
             */
                
                if(weights[0] != nullptr && phase[0] != nullptr){

                    MPI_Send(weights[0], dims_basis[0], mpi_precision, status.MPI_SOURCE, mpi_cmds::kill, MPI_COMM_WORLD);
                    MPI_Send(phase[0], sizeof_vector(dims_phase_per_fried), mpi_precision, status.MPI_SOURCE, mpi_cmds::kill, MPI_COMM_WORLD);
                
                }else{

                    fprintf(console, "(Error)\tNull buffer in MPI_Send(), calling MPI_Abort()\n");
                    fflush (console);
                
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                }

            /* -------------------------
             * Decrement processes_total
             * -------------------------
             */

                processes_total--;

            }

        }

    /* -----------------------------------------
     * !(8) Save residual phase-screens to disk.
     * -----------------------------------------
     */

        fprintf(console, "\n(Info)\tWriting to file:\t[%s, ", config::write_residual_to.c_str());
        fflush (console);
        
        write_status = residual.wr_fits(config::write_residual_to.c_str(), config::output_clobber);
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
     * Variable declaration.
     * ----------------------------------------
     * Name                 Type    Description
     * ----------------------------------------
     * dims_basis_naxis     sizt    Number of axes of dims_basis.
     */

        sizt dims_basis_naxis;

    /* ------------------------------------------------------------
     * Get the number of dimensions of basis functions from master.
     * ------------------------------------------------------------
     */

        MPI_Recv(&dims_basis_naxis, 1, MPI::UNSIGNED_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    /*
     * Vector declaration.
     * ----------------------------------------------------
     * Name                     Type            Description
     * ----------------------------------------------------
     * dims_phase_per_fried     sizt_vector     Dimensions of phase-screen simulations, per fried.
     * dims_phase_single        sizt_vector     Dimensions of a single phase-screen.
     * dims_weights             sizt_vector     Dimensions of the basis weights.
     * dims_basis               sizt_vector     Dimensions of basis functions.    
     */

        sizt_vector dims_phase_per_fried{config::sims_per_fried, config::sims_size_x, config::sims_size_y};
        sizt_vector dims_phase_single{config::sims_size_x, config::sims_size_y};
        sizt_vector dims_basis(dims_basis_naxis);
        sizt_vector dims_weights(1);
        
        MPI_Recv(dims_basis.data(), dims_basis_naxis, MPI::UNSIGNED_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    /*
     * Array declaration.
     * --------------------------------------------
     * Name         Type                Description
     * --------------------------------------------
     * phase        Array<precision>    Phase-screen simulations.
     * basis        Array<precision>    Basis functions.
     * weights      Array<precision>    Basis weights.
     */

        dims_weights[0] = dims_basis[0];

        Array<precision> phase(dims_phase_per_fried);
        Array<precision> basis(dims_basis);
        Array<precision> weights(dims_weights);

    /*
     * Vector declaration.
     * ----------------------------------------
     * Name         Type            Description
     * ----------------------------------------
     * basis_norm   sizt_vector     L2 normalization of the basis functions.
     */

        sizt_vector basis_norm(dims_basis[0]);

    /* ------------------------------------------------------------------------
     * Get basis functions from master rank, if this process hasn't been killed.
     * -------------------------------------------------------------------------
     */

        if(status.MPI_TAG != mpi_cmds::kill){

            MPI_Recv(basis[0], basis.get_size(), mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        /* ------------------------------------------------
        * Compute L2 normalization of the basis functions.
        * ------------------------------------------------
        */

            for(sizt ind = 0; ind < dims_basis[0]; ind++){
                for(sizt xs = 0; xs < dims_basis[1]; xs++){
                    for(sizt ys = 0; ys < dims_basis[2]; ys++){
                        basis_norm[ind] += basis(ind, xs, ys) * basis(ind, xs, ys);
                    }
                }
            }
        }

    /* --------------------------------------------
     * Compute residual phase-screens until killed.
     * --------------------------------------------
     */

        while(status.MPI_TAG != mpi_cmds::kill){

        /* ------------------------------
         * Get basis weights from master.
         * ------------------------------
         */

            MPI_Recv(weights[0], dims_weights[0], mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        /* -----------------------------------------
         * Get phase-screen simulations from master.
         * -----------------------------------------
         */

            MPI_Recv(phase[0], phase.get_size(), mpi_precision, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if(status.MPI_TAG == mpi_cmds::task){

            /* ---------------------------------------------
             * Compute the residuals from the phase-screens.
             * ---------------------------------------------
             */
                Array<precision> phase_single(dims_phase_single);
                for(sizt sim = 0; sim < dims_phase_per_fried[0]; sim++){

                    memcpy(phase_single[0], phase[sim], phase_single.get_size()*sizeof(precision));
                    make_residual_phase_screen(phase_single, basis, weights, basis_norm);
                    memcpy(phase[sim], phase_single[0], phase_single.get_size()*sizeof(precision));

                }

                if(phase[0] != nullptr){

                    MPI_Send(phase[0], phase.get_size(), mpi_precision, 0, mpi_pmsg::ready, MPI_COMM_WORLD);

                }else{

                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

                }
            }
        }
    }

    MPI_Finalize();
    return(EXIT_SUCCESS);
}
