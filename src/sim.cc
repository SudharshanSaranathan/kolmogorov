#include "mpi.h"
#include "libmp.h"
#include "fftw3.h"
#include "config.h"
#include "libfits.h"
#include "libarray.h"
#include "simulator.h"

#include <ctime>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <iostream>

int main(int argc, char *argv[]){

  MPI_Status status;
  int ProcessRank = 0;
  int ProcessTotal = 0;
  int ProcessActive = 0;
  int MPI_Recv_Count = 0;
  int ProcessDeleted = 0;

  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &ProcessTotal);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcessRank);

  FILE *console   = ProcessRank == 0 ? stdout : fopen("/dev/null","wb");
  fprintf(console, "------------------------------------------------------\n");
  fprintf(console, "- Turbulence-degraded phasescreen simulation program -\n");
  fprintf(console, "------------------------------------------------------\n");

  if(argc < 2){
    fprintf(console, "- Config file required. Aborting!\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }


  fprintf(console, "- Parsing %s: ", argv[1]); fflush(console);
  if(config::parse(argv[1]) == EXIT_FAILURE){
    fprintf(console, "%s not found\n", argv[1]);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  fprintf(console, "\t\tDone\n");
  if(ProcessRank == 0){

    long    NextIndex = 0;
    long    ProgressCounter = 0;
    double  ProgressPercentage = 0.0;

    array<double> Pupil;
    array<double> FriedParameter;

//  Read the fried parameter file.
    fprintf(console, "- Reading %s: ", config::Pupil.c_str()); fflush(console);
    if(fitsio<double>::read(config::Pupil.c_str(), &Pupil) == EXIT_FAILURE){
      fprintf(console, "\t\tFailed\n"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    fprintf(console, "\t\tDone\n");

//  Read the pupil file.
    fprintf(console, "- Reading %s: ", config::Fried.c_str()); fflush(console);
    if(fitsio<double>::read(config::Fried.c_str(), &FriedParameter) == EXIT_FAILURE){
      fprintf(console, "\t\tFailed\n"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    fprintf(console, "\t\tDone [%ld values]\n", FriedParameter.nelements());
    fprintf(console, "- Initializing processes: "); fflush(console);

//  Initialize arrays for storing phase-screens.
    vector_t DimsPerFried{config::Realizations, config::XDims, config::YDims};
    vector_t DimsPhasescreens{FriedParameter.nelements(), config::Realizations, config::XDims, config::YDims};
    vector_t PIDIndexMap; PIDIndexMap.resize(FriedParameter.nelements()+1);
    array<double> Phasescreens(DimsPhasescreens);

    for(int pid = 1; pid < ProcessTotal; pid++){
      if(pid > FriedParameter.nelements()){
        MPI_Send(FriedParameter.data, 1, MPI_DOUBLE, pid, CMDS::SHUTDOWN, MPI_COMM_WORLD);
        MPI_Send(Pupil.data, Pupil.nelements(), MPI_DOUBLE, pid, CMDS::SHUTDOWN, MPI_COMM_WORLD);
        ProcessTotal--;
      }else{
        MPI_Send(FriedParameter.data+NextIndex, 1, MPI_DOUBLE, pid, CMDS::KEEPALIVE, MPI_COMM_WORLD);
        MPI_Send(Pupil.data, Pupil.nelements(), MPI_DOUBLE, pid, CMDS::KEEPALIVE, MPI_COMM_WORLD);
        PIDIndexMap[pid] = NextIndex;
        NextIndex++;
      }
    }
    fprintf(console, "\tDone [%d MPI processes]\n", ProcessTotal);
    fprintf(stdout, "\r- Simulating phasescreens: \tIn Progress [%0.3lf %%]", ProgressPercentage);
    fflush(console);

    while(ProgressCounter < FriedParameter.nelements()){
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &MPI_Recv_Count);

      long MPI_Recv_loc = PIDIndexMap[status.MPI_SOURCE];
      MPI_Recv(Phasescreens.data4D[MPI_Recv_loc][0][0], MPI_Recv_Count,\
               MPI_DOUBLE, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,\
               &status);
      ProgressCounter++;
      ProgressPercentage = ProgressCounter*100.0/FriedParameter.nelements();
      fprintf(stdout, "\r- Simulating phasescreens: \tIn Progress [%0.3lf %%]", ProgressPercentage);
      fflush(console);
      if(NextIndex < FriedParameter.nelements()){
        MPI_Send(FriedParameter.data + NextIndex, 1, MPI_DOUBLE,\
                 status.MPI_SOURCE, CMDS::KEEPALIVE, MPI_COMM_WORLD);
        PIDIndexMap[status.MPI_SOURCE] = NextIndex;
        NextIndex++;
      }
    }
    fprintf(console, "\n- Shutting down processes: "); fflush(console);
    for(int pid=1; pid < ProcessTotal; pid++){
      MPI_Send(nullptr, 0, MPI_CHAR, pid, CMDS::SHUTDOWN, MPI_COMM_WORLD);
    }
    fprintf(console, "\tDone\n");
    fprintf(console, "- Writing to file: "); fflush(console);
    fitsio<double>::write(config::Outfile.c_str(), &Phasescreens, true);
    fprintf(console, "\t\tDone\n");

  }else if(ProcessRank){

    vector_t DimsSimulation{ulong(config::Simsize/config::Aperture)*config::XDims,\
                    ulong(config::Simsize/config::Aperture)*config::YDims};
    vector_t DimsAperture{config::XDims, config::YDims};
    vector_t DimsPerFried{config::Realizations, DimsAperture[0], DimsAperture[1]};

    array<double>     Pupil(DimsAperture);
    array<double>     PhasescreenPerFried(DimsPerFried);
    array<complex_t>  PhaseScreenSingle(DimsSimulation);
    array<complex_t>  PhasescreenSingleFourier(DimsSimulation);

    fftw_import_wisdom_from_filename(config::FFTWisdom.c_str());
    fftw_plan Reverse = fftw_plan_dft_2d(DimsSimulation[0], DimsSimulation[1],\
                                         reinterpret_cast<fftw_complex*>(PhasescreenSingleFourier.data),\
                                         reinterpret_cast<fftw_complex*>(PhaseScreenSingle.data),\
                                         FFTW_BACKWARD, FFTW_MEASURE);

    double CutoffRadius   = config::XDims/(2.0*config::Sampling);
    double FriedParameter = 0.0;

    uint XArrSize  = uint(config::XDims/2.);
    uint YArrSize  = uint(config::YDims/2.);
    uint XSimSize  = uint(DimsSimulation[0]/2.);
    uint YSimSize  = uint(DimsSimulation[0]/2.);
    long nelements = PhaseScreenSingle.nelements();
//    CreatePupil(&Pupil, CutoffRadius);
//    if(access("pupil.fits", F_OK)){
//      fitsio<double>::write("pupil.fits", &Pupil, true);
//    }

    MPI_Recv(&FriedParameter, 1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(Pupil.data, Pupil.nelements(), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    while(status.MPI_TAG != CMDS::SHUTDOWN){
      for(int l=0; l<config::Realizations; l++){
        double total = 0.0;
        SimulatePhaseFourier(&PhasescreenSingleFourier, FriedParameter, config::Sampling*config::Simsize);
        fftw_execute_dft(Reverse, reinterpret_cast<fftw_complex*>(PhasescreenSingleFourier.data),\
                                  reinterpret_cast<fftw_complex*>(PhaseScreenSingle.data));

        for(int i = XSimSize-XArrSize; i < XSimSize-XArrSize+config::XDims; i++){
          for(int j = YSimSize-YArrSize; j < YSimSize-YArrSize+config::YDims; j++){
            PhasescreenPerFried.data3D[l][i-(XSimSize-XArrSize)][j-(YSimSize-YArrSize)] =\
            Pupil.data2D[i-(XSimSize-XArrSize)][j-(YSimSize-YArrSize)]*PhaseScreenSingle.data2D[i][j].real();
          }
        }
      }
      MPI_Send(PhasescreenPerFried.data, PhasescreenPerFried.nelements(), MPI_DOUBLE, 0, PMSG::READY, MPI_COMM_WORLD);
      MPI_Recv(&FriedParameter, 1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
    fftw_destroy_plan(Reverse);
    fftw_cleanup();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return(EXIT_SUCCESS);
}
