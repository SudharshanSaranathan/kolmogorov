GCC    =  mpicc
MPICC  =  mpicc
CSTD   = -std=c++11 -Wall

INCDIR = ../include
OBJDIR = ../lib
BINDIR = ../bin

IFLAGS = -I${INCDIR} -I/usr/include
LFLAGS = -L/usr/lib -L/usr/lib64/
OFLAGS = -Ofast
LIBS   = -lm -lstdc++ -lcfitsio -lfftw3 -lmpi -lm -lpthread -lrt

DEPS_  = config.h lib_array.h lib_mem.h lib_phase.h
OBJS_  = config.o lib_array.o test_lib_mem.o test_lib_array.o

DEPS   = ${patsubst %,${INCDIR}/%,${DEPS_}}
OBJS   = ${patsubst %,${OBJDIR}/%,${OBJS_}}

LIB_TEST_OBJS_   = lib_array.o test_lib_array.o
MEM_TEST_OBJS_   = test_lib_mem.o

MAKE_PHS_OBJS_ = config.o lib_array.o make_phase.o
MAKE_RES_OBJS_ = config.o lib_array.o make_residual.o
MAKE_PSF_OBJS_ = config.o lib_array.o make_psf.o
MAKE_IMG_OBJS_ = config.o lib_array.o make_img.o

LIB_TEST_OBJS   = ${patsubst %,${OBJDIR}/%,${LIB_TEST_OBJS_}}
MEM_TEST_OBJS   = ${patsubst %,${OBJDIR}/%,${MEM_TEST_OBJS_}}

MAKE_PHS_OBJS = ${patsubst %,${OBJDIR}/%,${MAKE_PHS_OBJS_}}
MAKE_RES_OBJS = ${patsubst %,${OBJDIR}/%,${MAKE_RES_OBJS_}}
MAKE_PSF_OBJS = ${patsubst %,${OBJDIR}/%,${MAKE_PSF_OBJS_}}
MAKE_IMG_OBJS = ${patsubst %,${OBJDIR}/%,${MAKE_IMG_OBJS_}}

.PHONY: all

${OBJDIR}/%.o: %.cc ${DEPS}
	${GCC} ${CSTD} -c -o $@ $< ${IFLAGS} ${OFLAGS} ${LIBS}

test_lib_mem: ${MEM_TEST_OBJS}
	${GCC} ${CSTD} -o ${BINDIR}/$@ $< ${IFLAGS} ${LFLAGS} ${OFLAGS} ${LIBS}

test_lib_array: ${LIB_TEST_OBJS}
	${GCC} ${CSTD} -o ${BINDIR}/$@ $^ ${IFLAGS} ${LFLAGS} ${OFLAGS} ${LIBS}

make_phase: ${MAKE_PHS_OBJS}
	${MPICC} ${CSTD} -o ${BINDIR}/$@ $^ ${IFLAGS} ${LFLAGS} ${OFLAGS} ${LIBS}

make_residual: ${MAKE_RES_OBJS}
	${MPICC} ${CSTD} -o ${BINDIR}/$@ $^ ${IFLAGS} ${LFLAGS} ${OFLAGS} ${LIBS}

make_psf: ${MAKE_PSF_OBJS}
	${MPICC} ${CSTD} -o ${BINDIR}/$@ $^ ${IFLAGS} ${LFLAGS} ${OFLAGS} ${LIBS}

make_img: ${MAKE_IMG_OBJS}
	${MPICC} ${CSTD} -o ${BINDIR}/$@ $^ ${IFLAGS} ${LFLAGS} ${OFLAGS} ${LIBS}

clean:
	rm -rf ${OBJS}
	rm -rf ${BINDIR}/test_lib_mem
	rm -rf ${BINDIR}/test_lib_array	
	rm -rf ${BINDIR}/make_phase
	rm -rf ${BINDIR}/make_residual
	rm -rf ${BINDIR}/make_psf
	rm -rf ${BINDIR}/make_img
	
	rm -rf ${LIB_TEST_OBJS}
	rm -rf ${MEM_TEST_OBJS}
	rm -rf ${MAKE_PHASE_OBJS}
	rm -rf ${MAKE_RES_OBJS}
	rm -rf ${MAKE_PSF_OBJS}
