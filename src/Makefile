CXX = mpicxx
#CXXFLAGS = -Wall -g -std=c++11 -O0 -fopenmp
#CXXFLAGS = -Wall -std=c++11 -I /usr/include/eigen3/ -I ./tiny-dnn -O3 -fopenmp
CXXFLAGS = -Wall -std=c++11 -I /usr/include/eigen3/ -I ./tiny-dnn -O3 -fopenmp
#CXXFLAGS = -Wall -std=c++11 -I /cluster/home/aleonel/eigen3 -I ./tiny-dnn -O3
VARS= -DSKILL_TO_EAT -DSEP_FOOD -DIMMORTALS #-DEPOCHS=200#-DIMMORTALS #-DSEP_FOOD#-DNMUTATE #-DDEBUG#-DIMMORTALS # -DNMUTATE_BRAIN  -DDEBUG#-DNMUTATE_BRAIN # -DINVISIBLE_FOOD -DINTERACT -DFILTER_STATIC_PERCEPTIONS #-DNMUTATE_BRAIN -DLEARN -DSKILL_TO_SEE -DFOOD_LOCK_SKILL -DSKILL_TO_EAT #-DVISUALIZE # -DDEBUG# -DSKILL_TO_EAT -DINVISIBLE_FOOD -DDEBUG
ODIR=obj
RL=rl
OUTNAME ?= dummy

_OBJ = population.o population_io.o agent.o field.o main.o genome.o
_OBJ_test = test.o genome.o
_OBJ_pql = pq_rl_logic.o
_OBJ_rql = rq_rl_logic.o rbm.o
_OBJ_ql = rl_logic.o
_OBJ_drl = dq_rl_logic.o
_OBJ_test_parallel = parallel.o

OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
OBJ_pql = $(patsubst %,$(ODIR)/%,$(_OBJ_pql))
OBJ_rql = $(patsubst %,$(ODIR)/%,$(_OBJ_rql))
OBJ_ql = $(patsubst %,$(ODIR)/%,$(_OBJ_ql))
OBJ_drl = $(patsubst %,$(ODIR)/%,$(_OBJ_drl))
OBJ_test = $(patsubst %,$(ODIR)/%,$(_OBJ_test))
OBJ_test_parallel = $(patsubst %,$(ODIR)/%,$(_OBJ_test_parallel))

RUN_ARGS = 20 10 20 5 10 1.0 100 200 1000 1 100 1 0 0 1.0 0 0.1 0.0

#IMPORTANT: Uncompatible object files due to different VARS make clean if changing binary
all: sim_rql #sim_rbm #test_parallel

#sim: population.o agent.o genome.o field.o
#	$(CXX) $(CXXFLAGS) -o $@ $^

#sim2: sim2.o population.o agent.o genome.o field.o
#	$(CXX) $(CXXFLAGS) $^ -o $@

sim_ql: VARS+=-DLEARN -DBRAIN_QL
sim_pql: VARS+=-DLEARN -DBRAIN_PQL
sim_rql: VARS+=-DLEARN -DBRAIN_RQL
sim_drl: VARS+=-DLEARN -DBRAIN_DEEP
test_drl: VARS+=-DTEST -DLEARN -DBRAIN_DEEP
test_pql: VARS+=-DTEST -DLEARN -DBRAIN_PQL
test_rql: VARS+=-DTEST -DLEARN -DBRAIN_RQL
test_ql: VARS+=-DTEST -DLEARN -DBRAIN_QL
#sim_drl: CXXFLAGS+=-pthread

special_ql: sim_ql
	cp sim_ql $(OUTNAME)

special_rql: sim_rql
	cp ./sim_rql $(OUTNAME)

special_drl: sim_drl
	cp sim_drl $(OUTNAME)

special_pql: sim_pql
	cp sim_pql $(OUTNAME)

special_nolearn: sim_pql
	cp sim_pql $(OUTNAME)

# sim: $(OBJ)
#		$(CXX) $(CXXFLAGS) $(VARS) $^ -o $@

sim_ql: $(OBJ) $(OBJ_ql)
	$(CXX) $(CXXFLAGS) $(VARS) $^ -o $@

sim_pql: $(OBJ) $(OBJ_pql)
	$(CXX) $(CXXFLAGS) $(VARS) $^ -o $@

sim_rql: $(OBJ) $(OBJ_rql)
	$(CXX) $(CXXFLAGS) $(VARS) $^ -o $@

sim_drl: $(OBJ) $(OBJ_drl)
	$(CXX) $(CXXFLAGS) $(VARS) $^ -o $@

$(ODIR)/%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) $(VARS) -o $@ -c $<

$(ODIR)/main.o: main.cpp
		$(CXX) $(CXXFLAGS) $(VARS) -o $@ -c $<

$(ODIR)/test.o: test.cpp
	$(CXX) $(CXXFLAGS) $(VARS) -o $@ -c $<

$(ODIR)/population_io.o: population_io.cpp population.hpp
	$(CXX) $(CXXFLAGS) $(VARS) -o $@ -c $<

$(ODIR)/parallel.o: parallel.cpp
	$(CXX) $(CXXFLAGS) $(VARS) -o $@ -c $<

run_ql:
	./sim_ql $(RUN_ARGS)

run_pql:
	./sim_pql $(RUN_ARGS)

run_rql:
	./sim_rql $(RUN_ARGS)

test_drl: $(OBJ_test) $(OBJ_drl)
	$(CXX) $(CXXFLAGS) $(VARS) $^ -o $@

test_pql: $(OBJ_test) $(OBJ_pql)
	$(CXX) $(CXXFLAGS) $(VARS) $^ -o $@

test_rql: $(OBJ_test) $(OBJ_rql)
	$(CXX) $(CXXFLAGS) $(VARS) $^ -o $@

test_ql: $(OBJ_test) $(OBJ_ql)
	$(CXX) $(CXXFLAGS) $(VARS) $^ -o $@

test_parallel: $(OBJ_test_parallel)
	$(CXX) $(CXXFLAGS) $(VARS) $^ -o $@
	cp test_parallel ../exec/

run_test_drl:
	./test_drl

run_test_rbm:
	./test_rbm

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o sim sim_* test_* dummy
