CC     =  mpicc
CSTD   = -std=c++11

INCDIR = ../include
OBJDIR = ../lib
BINDIR = ../bin

IFLAGS = -I${INCDIR} -I/opt/local/include
LFLAGS = -L/opt/local/lib64 -L/opt/local/lib
OFLAGS = -O3 -ffast-math
LIBS   = -lm -lstdc++ -lcfitsio -lfftw3 -lmpi -lm -lpthread -lrt

DEPS_  = config.h lib_array.h lib_mem.h lib_phase.h
OBJS_  = config.o lib_array.o test_lib_mem.o test_lib_array.o

DEPS   = ${patsubst %,${INCDIR}/%,${DEPS_}}
OBJS   = ${patsubst %,${OBJDIR}/%,${OBJS_}}

LIBTESTOBJS_   = lib_array.o test_lib_array.o
MEMTESTOBJS_   = test_lib_mem.o
MAKEPHASEOBJS_ = config.o lib_array.o make_phase.o
MAKERESDLOBJS_ = config.o lib_array.o residual.o

LIBTESTOBJS    = ${patsubst %,${OBJDIR}/%,${LIBTESTOBJS_}}
MEMTESTOBJS    = ${patsubst %,${OBJDIR}/%,${MEMTESTOBJS_}}
MAKEPHASEOBJS  = ${patsubst %,${OBJDIR}/%,${MAKEPHASEOBJS_}}
MAKERESDLOBJS  = ${patsubst %,${OBJDIR}/%,${MAKERESDLOBJS_}}


.PHONY: all

${OBJDIR}/%.o: %.cc ${DEPS}
	${CC} ${CSTD} -c -o $@ $< ${IFLAGS} ${OFLAGS} ${LIBS}

test_lib_mem: ${MEMTESTOBJS}
	${CC} ${CSTD} -o ${BINDIR}/$@ $< ${IFLAGS} ${LFLAGS} ${OFLAGS} ${LIBS}

test_lib_array: ${LIBTESTOBJS}
	${CC} ${CSTD} -o ${BINDIR}/$@ $^ ${IFLAGS} ${LFLAGS} ${OFLAGS} ${LIBS}

make_phase: ${MAKEPHASEOBJS}
	${CC} ${CSTD} -o ${BINDIR}/$@ $^ ${IFLAGS} ${LFLAGS} ${OFLAGS} ${LIBS}

residual: ${MAKERESDLOBJS}
	${CC} ${CSTD} -o ${BINDIR}/$@ $^ ${IFLAGS} ${LFLAGS} ${OFLAGS} ${LIBS}

clean:
	rm -rf ${OBJS}
	rm -rf ${BINDIR}/memtest ${BINDIR}/arraytest ${BINDIR}/make_phase
