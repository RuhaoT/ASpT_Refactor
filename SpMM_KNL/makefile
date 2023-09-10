include ../misc/Makefile.in
LIBDIR:=../$(LIBDIR)
NRMCL=$(LIBDIR)/$(LIBNAME)
LDFLAGS=$(NRMCL) -qopenmp $(LDOPTIONS)

CFLAGS = $(COPTIONS) $(OPTFLAGS) $(CINCLUDES) -O3 -parallel -dynamic -qopenmp -Ofast -xMIC-AVX512 -simd -vec -qopt-prefetch=5  -qopt-prefetch-distance=0,4  
CFLAGS += -I$(LIBDIR) -L.

SOURCES=$(wildcard *.cc)
OBJS=$(shell echo $(SOURCES) | sed s/.cc/.o/g)
EXES=$(shell echo $(SOURCES) | sed s/.cc/.x/g)

all:$(EXES)

CC=icpc
%.x:%.o $(NRMCL)
	@echo $(LDFLAGS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(NRMCL):$(shell find $(LIBDIR) -name '*.cc' -o -name '*.h')
	(cd $(LIBDIR); make)

%.o:%.cc
	$(CC) $(CFLAGS) $^ -c

clean:
	rm -rf *.o *.x

diskclean:
	rm -rf *.o *.x
	make -C $(LIBDIR) clean

test:
	for exe in $(shell ls *.x); \
	do \
		echo $${exe}; \
		./$${exe}; \
	done


#icpc -m64  -w -I"/opt/intel/composer_xe_2013.4.183/mkl/include"         ./source/nqp.cc -Wl,--start-group         "/opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_intel_lp64.a"         "/opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_intel_thread.a"         "/opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_core.a"         -Wl,--end-group -L"/opt/intel/composer_xe_2013.4.183/mkl/../compiler/lib/intel64" -liomp5 -lpthread -lm -ldl -o _results/intel_lp64_parallel_intel64_lib/nqp.out
