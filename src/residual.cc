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

/*
 * Description:
 * This program subtracts basis functions defined on the aperture from phase-screen simulations.
 * The subtraction is weighted, if weights are provided. The residual phase-screens represent
 * corrections to the phase-screens either by Adaptive Optics (AO) or by post-facto image
 * processing. 
 *
 * Usage:
 * mpiexec -np <cores>> ./make_phase <config_file>
 *
 * Inputs:
 * See config.h for a detailed explanation of the inputs to the program.
 *
 * Outputs:
 * See config.h for a detailed explanation of the output of the program.
 *
 * Program logic: 
 * 1. Parse config file.
 * 2. Read phase-screens from file.
 * 3. Read basis functions from file.
 * 4. Read basis weights from file.
 * 5. Distribute phase-screens to workers.
 * 6. Get residuals from workers, store in memory.
 * 7. Repeat steps 5-6 for all phase-screens.
 * 8. Save residuals to disk.
 *
 * Additional information:
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

/*
 * Initialize MPI
 */
   
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processes_total);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

/*
 * Only the master MPI process - rank zero - prints to stdout.
 */

    FILE *console   = process_rank == 0 ? stdout : fopen("/dev/null","wb");
    fprintf(console, "------------------------------------------------------\n");
    fprintf(console, "- Phase-screen residuals computation program -\n");
    fprintf(console, "------------------------------------------------------\n");

/*
 * (!) Read and parse config file. 
 */

    if(argc < 2){
	    fprintf(console, "(Error)\tExpected configuration file, aborting!\n");
	    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    fprintf(console, "(Info)\tReading configuration:\t\t");
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
     * residual Array<double>   Residual phase-screens.
     */

        Array<double> phase;
        Array<double> basis;
        Array<double> weights;

    /*
     * Read phase-screen simulations.
     */

        fprintf(console, "(Info)\tReading %s:\t", config::write_phase_to.c_str());
        read_status = phase.rd_fits(config::write_phase_to.c_str());
        if(read_status != EXIT_SUCCESS){
            fprintf(console, "[Failed, Err code: %d]\n", read_status);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }else{
            fprintf(console, "[Done]\n");
        }

    /*
     * Read basis functions.
     */

        fprintf(console, "(Info)\tReading %s:\t", config::read_basis_functions_from.c_str());
        read_status = basis.rd_fits(config::read_basis_functions_from.c_str());
        if(read_status != EXIT_SUCCESS){
            fprintf(console, "[Failed, Err code: %d]\n", read_status);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }else{
            fprintf(console, "[Done]\n");
        }

    /*
     * Read basis weights.
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
     * process_fried_map    std::vector<sizt>   Map of which process is handling which fried index.
     */

        const sizt_vector dims_phase   = phase.get_dims();
        const sizt_vector dims_basis   = basis.get_dims();
        const sizt_vector dims_weights = weights.get_dims();
        sizt_vector process_fried_map(dims_phase[0] + 1);
        
    }else if(process_rank){



    }

    MPI_Finalize();
    return(EXIT_SUCCESS);
}
