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
 * mpiexec -np <cores>> ./make_phase <config_file>
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
 */
   
    MPI_Status status;
    int process_rank = 0;
    int processes_total = 0;
    int mpi_recv_count = 0;
    int read_status = 0;

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
     * progress_percent         double  Progress percentage.
     */

        long    index_of_fried_in_queue = 0;
        long    fried_completed = 0;
        double  progress_percent = 0.0;

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

        fprintf(console, "(Info)\tReading %s:\t", config::write_phase_to.c_str());
        read_status = phase.rd_fits(config::write_phase_to.c_str());
        if(read_status != EXIT_SUCCESS){
            fprintf(console, "[Failed, Err code: %d]\n", read_status);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }else{
            fprintf(console, "[Done]\n");
        }

    /* -------------------------------
     * Read basis functions from file.
     * -------------------------------
     */

        fprintf(console, "(Info)\tReading %s:\t", config::read_basis_from.c_str());
        read_status = basis.rd_fits(config::read_basis_from.c_str());
        if(read_status != EXIT_SUCCESS){
            fprintf(console, "[Failed, Err code: %d]\n", read_status);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }else{
            fprintf(console, "[Done]\n");
        }

    /* -----------------------------
     * Read basis weights from file.
     * -----------------------------
     */

        fprintf(console, "(Info)\tReading %s:\t", config::read_weights_from.c_str());
        read_status = weights.rd_fits(config::read_weights_from.c_str());
        if(read_status != EXIT_SUCCESS){
            fprintf(console, "[Failed, Err code: %d]\n", read_status);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }else{
            fprintf(console, "[Done]\n");
        }

    /*
     * Vector declaration.
     * ----------------------------------------------------
     * Name                 Type                Description
     * ----------------------------------------------------
     * dims_phase           std::vector<sizt>   Dimensions of phase-screen simulations.
     * dims_basis           std::vector<sizt>   Dimensions of basis functions.
     * dims_weights         std::vector<sizt>   Dimensions of basis weights.
     * dims_phase_per_fried std::vector<sizt>   Dimensions of phase-screen simulations, per fried.
     * process_fried_map    std::vector<sizt>   Map of which process is handling which fried index.
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

        if(dims_phase[0] != dims_weights[0]){
            fprintf(console, "(Error)\tExpected equal dimensions for phase-screens and weights. Aborting\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }else if(dims_phase[2] != dims_basis[1] || dims_phase[3] != dims_basis[2]){
            fprintf(console, "(Error)\tExpected equal dimensions for phase-screens and basis functions. Aborting\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }else if(dims_weights[1] != dims_basis[0]){
            fprintf(console, "(Error)\tExpected equal dimensions for basis functions and basis weights. Aborting\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

    /*
     * Array declaration.
     * ----------------------------------------
     * Name         Type            Description
     * ----------------------------------------
     * residual     Array<double>   Residual phase-screens, see 'lib_array.h' for datatype.
     * basis_normed Array<double>   L2 normalized basis functions, see 'lib_array.h' for datatype.
     */

        Array<double> residual(phase);
        Array<double> basis_normed(basis);

    /*
     * Variable declaration.
     * ----------------------------------------
     * Name                 Type    Description
     * ----------------------------------------
     * dims_basis_naxis     sizt    Number of axes of dims_basis.
     * id                   sizt    Rank of MPI process.
     */

        sizt dims_basis_naxis = dims_basis.size();

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

            MPI_Send(basis.root_ptr, basis.get_size(), MPI_DOUBLE, id, mpi_cmds::stayalive, MPI_COMM_WORLD);

        }

    /* --------------------------------------------------
     * Begin sending phase-screen simulations to workers.
     * --------------------------------------------------
     */

        for(int id = 1; id < processes_total; id++){

        /*
	     * Variable declaration:
	     *-----------------------------------------
	     * Name		            Type    Description
	     * ----------------------------------------
	     * fried_index_weights	sizt	Index of next fried in pointer space, for array weights.
         * fried_index_phase    sizt    Index of next fried in pointer space, for array phase.
	     */

            sizt fried_index_weights = index_of_fried_in_queue * dims_basis[0];
            sizt fried_index_phase   = index_of_fried_in_queue * sizeof_vector(dims_phase_per_fried);

            MPI_Send(weights.root_ptr + fried_index_weights, dims_basis[0], MPI_DOUBLE, id, mpi_cmds::stayalive, MPI_COMM_WORLD);
            MPI_Send(phase.root_ptr + fried_index_phase, sizeof_vector(dims_phase_per_fried), MPI_DOUBLE, id, mpi_cmds::stayalive, MPI_COMM_WORLD);

            process_fried_map[id] = index_of_fried_in_queue;
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

            sizt fried_index_weights = process_fried_map[status.MPI_SOURCE] * dims_basis[0];
	        sizt fried_index_phase   = process_fried_map[status.MPI_SOURCE] * sizeof_vector(dims_phase_per_fried);

	    /* ------------------------------------------------------
	     * Get residual phase-screens, store at correct location.
         * ------------------------------------------------------
	     */

	        MPI_Recv(residual.root_ptr + fried_index_phase, sizeof_vector(dims_phase_per_fried), MPI_DOUBLE, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        /* --------------------------
	     * Increment fried_completed.
         * --------------------------
	     */

	        fried_completed++;

        /* -------------------------------------------------------------------
	     * Send next set of phase-screen simulations, if available, to worker.
         * -------------------------------------------------------------------
	     */

	        if(index_of_fried_in_queue < dims_phase[0]){

                fried_index_weights = index_of_fried_in_queue * dims_basis[0];
                fried_index_phase   = index_of_fried_in_queue * sizeof_vector(dims_phase_per_fried);

                MPI_Send(weights.root_ptr + fried_index_weights, dims_basis[0], MPI_DOUBLE, status.MPI_SOURCE, mpi_cmds::stayalive, MPI_COMM_WORLD);
		        MPI_Send(phase.root_ptr + index_of_fried_in_queue, sizeof_vector(dims_phase_per_fried), MPI_DOUBLE, status.MPI_SOURCE, mpi_cmds::stayalive, MPI_COMM_WORLD);
	
	        /* -----------------------------------------------------------------
	         * Update process_fried_map, and increment index_of_fried_in_queue).
             * ----------------------------------------------------------------- 
	         */
	
		        process_fried_map[status.MPI_SOURCE] = index_of_fried_in_queue;
		        index_of_fried_in_queue++;
      	    }
	        else{
	    
	        /* ----------------------------------------------------------
	         * If no more residuals to be calculated, shutdown processes.
             * ----------------------------------------------------------
	         */
                
                MPI_Send(weights.root_ptr, dims_basis[0], MPI_DOUBLE, status.MPI_SOURCE, mpi_cmds::shutdown, MPI_COMM_WORLD);
		        MPI_Send(phase.root_ptr, sizeof_vector(dims_phase_per_fried), MPI_DOUBLE, status.MPI_SOURCE, mpi_cmds::shutdown, MPI_COMM_WORLD);
	        
            }

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

        sizt        dims_basis_naxis;
        MPI_Recv(&dims_basis_naxis, 1, MPI_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    /*
     * Vector declaration.
     * ----------------------------------------------------
     * Name                     Type            Description
     * ----------------------------------------------------
     * dims_phase_per_fried     sizt_vector     Dimensions of phase-screen simulations, per fried.
     * dims_weights             sizt_vector     Dimensions of the basis weights.
     * dims_basis               sizt_vector     Dimensions of basis functions.    
     */

        sizt_vector dims_phase_per_fried{config::sims_per_fried, config::sims_size_x, config::sims_size_y};
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

        MPI_Recv(basis.root_ptr, basis.get_size(), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        while(status.MPI_TAG != mpi_cmds::shutdown){

            MPI_Recv(weights.root_ptr, dims_weights[0], MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(phase.root_ptr, phase.get_size(), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if(status.MPI_TAG == mpi_cmds::stayalive){

            /*
             * Compute the residual phase-screens for the given phase-screen simulations. 
             */
            
                MPI_Send(phase.root_ptr, phase.get_size(), MPI_DOUBLE, 0, mpi_pmsg::ready, MPI_COMM_WORLD);

            }
        }
    }

    MPI_Finalize();
    return(EXIT_SUCCESS);
}
