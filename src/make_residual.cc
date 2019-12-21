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
 * 1. Parse config file.
 * 2. Read phase-screens from file.
 * 3. Read basis functions from file.
 * 4. Read basis weights from file.
 * 5. Distribute phase-screens to workers.
 * 6. Get residuals from workers, store in memory.
 * 7. Repeat steps 5-6 for all phase-screens.
 * 8. Save residuals to disk.
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
 *  -----------------------------------------------
 *  Name		Type		Description
 *  -----------------------------------------------
 *  status:		MPI_status	See MPI documentation.
 *  process_rank:	int		Rank of MPI processes.
 *  process_total:	int		Store the total number of MPI processes
 *  mpi_recv_count:	int		Store the count of data received in MPI_Recv, see MPI documentation for explanation.
 *  read_status:	int		File read status.
 *  write_status:   int     File write status.
 */
   
    MPI_Status status;
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
    fprintf(console, "- Phase-screen residuals computation program -\n");
    fprintf(console, "------------------------------------------------------\n");

/* -------------------------------
 * !(1) Read and parse config file.
 * -------------------------------
 */

    if(argc < 2){
	    fprintf(console, "(Error)\tExpected configuration file, aborting!\n");
	    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    fprintf(console, "(Info)\tReading configuration:\t");
    if(config_parse(argv[1]) == EXIT_FAILURE){
	    fprintf(console, "[Failed]\n");
	    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }else{
	    fprintf(console, "[Done]\n");
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
     * index_of_fried_in_queue  long    Index of the next fried parameter.
     * fried_completed          long    Number of fried parameters processed.
     * percent_assigned         double  Percentage of fried assigned.
     * percent_completed        double  Percentage of fried completed.
     */

        long    index_of_fried_in_queue = 0;
        long    fried_completed         = 0;
        double  percent_assigned        = 0.0;
        double  percent_completed       = 0.0;

    /*
     * Array declaration.
     * ------------------------------------
     * Name     Type            Description
     * ------------------------------------
     * phase    Array<double>   Phase-screen simulations, see 'lib_array.h' for datatype.
     * basis    Array<double>   Basis functions on aperture.
     * weights  Array<double>   Basis weights.
     */

        Array<double> phase;
        Array<double> basis;
        Array<double> weights;

    /* ----------------------------------------
     * Read phase-screen simulations from file.
     * ----------------------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::write_phase_to.c_str());
        read_status = phase.rd_fits(config::write_phase_to.c_str());
        if(read_status != EXIT_SUCCESS){
            fprintf(console, "Failed with err code: %d]\n", read_status);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }else{
            fprintf(console, "Done]\n");
        }

    /* -------------------------------
     * Read basis functions from file.
     * -------------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::read_basis_from.c_str());
        read_status = basis.rd_fits(config::read_basis_from.c_str());
        if(read_status != EXIT_SUCCESS){
            fprintf(console, "Failed with err code: %d]\n", read_status);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }else{
            fprintf(console, "Done]\n");
        }

    /* -----------------------------
     * Read basis weights from file.
     * -----------------------------
     */

        fprintf(console, "(Info)\tReading file:\t\t[%s, ", config::read_weights_from.c_str());
        read_status = weights.rd_fits(config::read_weights_from.c_str());
        if(read_status != EXIT_SUCCESS){
            fprintf(console, "Failed with err code: %d]\n", read_status);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }else{
            fprintf(console, "Done]\n");
        }

    /*
     * Vector declaration.
     * ----------------------------------------------------
     * Name                 Type            Description
     * ----------------------------------------------------
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
     
            fprintf(console, "(Error)\texpected phase-screens with size [%ld %ld], calling MPI_Abort()\n", config::sims_size_x, config::sims_size_y);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

        }

        if(dims_basis[1] != config::sims_size_x  || dims_basis[2] != config::sims_size_y){
     
            fprintf(console, "(Error)\texpected basis functions with size [%ld %ld], calling MPI_Abort()\n", config::sims_size_x, config::sims_size_y);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

        }

        if(dims_weights[0] != dims_phase[0] || dims_weights[1] != dims_basis[0]){
            
            fprintf(console, "(Error)\texpected weights with dimensions [%ld %ld], calling MPI_Abort()\n", dims_phase[0], dims_basis[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    
        }

     /*
     * Array declaration.
     * ----------------------------------------
     * Name         Type            Description
     * ----------------------------------------
     * residual     Array<double>   Residual phase-screens, see 'lib_array.h' for datatype.
     */

        Array<double> residual(phase);

    /*
     * Variable declaration.
     * ----------------------------------------
     * Name                 Type    Description
     * ----------------------------------------
     * dims_basis_naxis     sizt    Number of axes of dims_basis.
     * id                   sizt    Rank of MPI process.
     */

        sizt dims_basis_naxis  = dims_basis.size();

        for(int id = 1; id < processes_total; id++){

        /* ------------------------------------------------------
         * If rank > number of fried parameters, shutdown worker.
         * ------------------------------------------------------
         */

            if(id > dims_phase[0]){

                MPI_Send(&dims_basis_naxis, 1, MPI_LONG, id, mpi_cmds::shutdown, MPI_COMM_WORLD);
                MPI_Send( dims_basis.data(), dims_basis_naxis, MPI_LONG, id, mpi_cmds::shutdown, MPI_COMM_WORLD);

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

                MPI_Send(&dims_basis_naxis, 1, MPI_LONG, id, mpi_cmds::stayalive, MPI_COMM_WORLD);
                MPI_Send( dims_basis.data(), dims_basis_naxis, MPI_LONG, id, mpi_cmds::stayalive, MPI_COMM_WORLD);

            }

        }

    /* --------------------------------
     * Send basis functions to workers.
     * --------------------------------
     */

        for(int id = 1; id < processes_total; id++){

            MPI_Send(basis[0], basis.get_size(), MPI_DOUBLE, id, mpi_cmds::stayalive, MPI_COMM_WORLD);

        }

    /* ----------------------
     * Loop over MPI workers.
     * ----------------------
     */

        for(int id = 1; id < processes_total; id++){

        /* -------------------
         * Send basis weights.
         * -------------------
         */

            if(weights[index_of_fried_in_queue] != nullptr){
                
                MPI_Send(weights[index_of_fried_in_queue], dims_basis[0], MPI_DOUBLE, id, mpi_cmds::stayalive, MPI_COMM_WORLD);

            }else{

                fprintf(console, "(Error)\tNull buffer, calling MPI_Abort()\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

            }

        /* -------------------
         * Send phase-screens.
         * -------------------
         */

            if(phase[index_of_fried_in_queue] != nullptr){
                
                MPI_Send(phase[index_of_fried_in_queue], sizeof_vector(dims_phase_per_fried), MPI_DOUBLE, id, mpi_cmds::stayalive, MPI_COMM_WORLD);

            }else{

                fprintf(console, "(Error)\tNull buffer, calling MPI_Abort()\n");
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
            fprintf(stdout, "\r(Info)\tComputing residuals: \t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
            fflush(console);

        /* ----------------------------------
         * Increment index_of_fried_in_queue.
         * ----------------------------------
         */

            index_of_fried_in_queue++;

        }

        while(fried_completed < dims_phase[0]){

        /* ----------------------------------------------------------------------------
	     * Wait for a worker that is ready. If found, get and store worker information.
         * ----------------------------------------------------------------------------
	     */	
	
	        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	        MPI_Get_count(&status, MPI_DOUBLE, &mpi_recv_count);

	    /*
	     * Variable declaration:
	     *-----------------------------------------
	     * Name		            Type    Description
	     * ----------------------------------------
	     * fried_index_weights	sizt	Index of next fried in pointer space, for array weights.
         * fried_index_phase    sizt    Index of next fried in pointer space, for array phase.
	     */

        /* -------------------------------------------------
	     * Get index of fried parameter processed by worker.
         * -------------------------------------------------
	     */

            sizt fried_index_weights = process_fried_map[status.MPI_SOURCE];
	        sizt fried_index_phase   = process_fried_map[status.MPI_SOURCE];

	    /* ------------------------------------------------------
	     * Get residual phase-screens, store at correct location.
         * ------------------------------------------------------
	     */

	        MPI_Recv(residual[fried_index_phase], sizeof_vector(dims_phase_per_fried), MPI_DOUBLE, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

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
            fprintf(stdout, "\r(Info)\tComputing residuals:\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
            fflush(console);

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
                
                MPI_Send(weights[index_of_fried_in_queue], dims_basis[0], MPI_DOUBLE, status.MPI_SOURCE, mpi_cmds::stayalive, MPI_COMM_WORLD);

            }else{

                fprintf(console, "(Error)\tNull buffer, calling MPI_Abort()\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

            }

        /* -------------------
         * Send phase-screens.
         * -------------------
         */

            if(phase[index_of_fried_in_queue] != nullptr){
                
                MPI_Send(phase[index_of_fried_in_queue], sizeof_vector(dims_phase_per_fried), MPI_DOUBLE, status.MPI_SOURCE, mpi_cmds::stayalive, MPI_COMM_WORLD);

            }else{

                fprintf(console, "(Error)\tNull buffer, calling MPI_Abort()\n");
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
                fprintf(stdout, "\r(Info)\tComputing residuals:\t[%0.1lf %% assigned, %0.1lf %% completed]", percent_assigned, percent_completed); 
                fflush(console);

            /* ----------------------------------
	         * Increment index_of_fried_in_queue.
             * ----------------------------------
	         */

		        index_of_fried_in_queue++;
      	    }
	        else{
	    
	        /* ----------------------------------------------------------
	         * If no more residuals to be calculated, shutdown processes.
             * ----------------------------------------------------------
	         */
                
                MPI_Send(weights[0], dims_basis[0], MPI_DOUBLE, status.MPI_SOURCE, mpi_cmds::shutdown, MPI_COMM_WORLD);
		        MPI_Send(phase[0], sizeof_vector(dims_phase_per_fried), MPI_DOUBLE, status.MPI_SOURCE, mpi_cmds::shutdown, MPI_COMM_WORLD);
                
            /* -------------------------
             * Decrement processes_total
             * -------------------------
             */

                processes_total--;

            }

        }

        fprintf(console, "\n(Info)\tWriting to file:\t[%s, ", config::write_residual_to.c_str()); fflush(console);
        write_status = residual.wr_fits(config::write_residual_to.c_str(), config::output_clobber);

	    if(write_status != EXIT_SUCCESS){
	        
            fprintf(console, "Failed with err code: %d]\n", write_status);
	    
        }
	    else{
	     
            fprintf(console, "Done]\n");
        
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
        MPI_Recv(&dims_basis_naxis, 1, MPI_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

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
        
        MPI_Recv(dims_basis.data(), dims_basis_naxis, MPI_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    /*
     * Array declaration.
     * ----------------------------------------
     * Name         Type            Description
     * ----------------------------------------
     * phase        Array<double>   Phase-screen simulations.
     * basis        Array<double>   Basis functions.
     * weights      Array<double>   Basis weights.
     */

        dims_weights[0] = dims_basis[0];

        Array<double> phase(dims_phase_per_fried);
        Array<double> basis(dims_basis);
        Array<double> weights(dims_weights);

    /* -------------------------------------
     * Get basis functions from master rank.
     * -------------------------------------
     */

        MPI_Recv(basis[0], basis.get_size(), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    /*
     * Vector declaration.
     * ----------------------------------------
     * Name         Type            Description
     * ----------------------------------------
     * basis_norm   Array<double>   L2 normalization of the basis functions.
     */

        sizt_vector basis_norm(dims_basis[0]);

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

    /* --------------------
     * Loop until shutdown.
     * --------------------
     */

        while(status.MPI_TAG != mpi_cmds::shutdown){

        /* ------------------------------
         * Get basis weights from master.
         * ------------------------------
         */

            MPI_Recv(weights[0], dims_weights[0], MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        /* -----------------------------------------
         * Get phase-screen simulations from master.
         * -----------------------------------------
         */

            MPI_Recv(phase[0], phase.get_size(), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if(status.MPI_TAG == mpi_cmds::stayalive){

            /* ---------------------------------------------
             * Compute the residuals from the phase-screens.
             * ---------------------------------------------
             */
                Array<double> phase_single(dims_phase_single);
                for(sizt sim = 0; sim < dims_phase_per_fried[0]; sim++){

                    memcpy(phase_single[0], phase[sim], phase_single.get_size()*sizeof(double));
                    make_residual_phase_screen(phase_single, basis, weights, basis_norm);
                    memcpy(phase[sim], phase_single[0], phase_single.get_size()*sizeof(double));

                }

                MPI_Send(phase[0], phase.get_size(), MPI_DOUBLE, 0, mpi_pmsg::ready, MPI_COMM_WORLD);

            }
        }
    }

    MPI_Finalize();
    return(EXIT_SUCCESS);
}