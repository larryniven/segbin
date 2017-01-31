CXXFLAGS += -std=c++11 -I .. -L ../speech -L ../nn -L ../autodiff -L ../opt -L ../la -L ../ebt -L ../seg -L ../fst

bin = \
    oracle-error \
    oracle-cost \
    learn-ctc \
    predict-ctc \
    learn-order1-full \
    predict-order1-full \
    prune-order1-full \
    learn-order1-lat \
    predict-order1-lat \
    forced-align-order1-full \
    prune-random \
    lat-order2-learn \
    lat-order2-predict \
    overlap-vs-per \
    segrnn-learn \
    segrnn-predict \
    segrnn-prune \
    segrnn-beam-prune \
    segrnn-align \
    segrnn-input-grad \
    segrnn-cascade-learn \
    segrnn-cascade-predict \
    segrnn-entropy-learn \
    segrnn-hypercolumn-learn

.PHONY: all clean

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

oracle-error: oracle-error.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

oracle-cost: oracle-cost.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-ctc: learn-ctc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-ctc: predict-ctc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-e2e-mll: learn-order1-e2e-mll.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-full: learn-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-full: predict-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

prune-order1-full: prune-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

forced-align-order1-full: forced-align-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-lat: learn-order1-lat.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-lat: predict-order1-lat.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

prune-random: prune-random.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

lat-order2-learn: lat-order2-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

lat-order2-predict: lat-order2-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

overlap-vs-per: overlap-vs-per.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-learn: segrnn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-predict: segrnn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-prune: segrnn-prune.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-beam-prune: segrnn-beam-prune.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-align: segrnn-align.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-cascade-learn: segrnn-cascade-learn.o cascade.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-cascade-predict: segrnn-cascade-predict.o cascade.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-input-grad: segrnn-input-grad.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-entropy-learn: segrnn-entropy-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-hypercolumn-learn: segrnn-hypercolumn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

