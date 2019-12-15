#ifndef _LIBMPI_
#define _LIBMPI_

#include "mpi.h"

typedef struct mpi_cmds{
    static int stayalive;
    static int shutdown;
} mpi_cmds;

typedef struct mpi_pmsg{
    static int ready;
    static int error;
    static int warning;
} mpi_pmsg;

typedef struct mpi_proc{
    static int total;
    static int active;
    static int killed;
} mpi_proc;

int mpi_cmds::stayalive = 2;
int mpi_cmds::shutdown  = 1;
int mpi_pmsg::ready     = 1;
int mpi_pmsg::error     = 2;
int mpi_pmsg::warning   = 3;
int mpi_proc::total     = 0;
int mpi_proc::active    = 0;
int mpi_proc::killed    = 0;

#endif
