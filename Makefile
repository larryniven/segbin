CXXFLAGS += -std=c++11 -I .. -L ../speech -L ../nn -L ../autodiff -L ../opt -L ../la -L ../ebt -L ../seg

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
    learn-order2-lat \
    predict-order2-lat \
    overlap-vs-per \
    segrnn-learn \
    segrnn-predict \
    segrnn-prune \
    segrnn-cascade-learn \
    segrnn-cascade-predict

.PHONY: all clean

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

oracle-error: oracle-error.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

oracle-cost: oracle-cost.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-ctc: learn-ctc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-ctc: predict-ctc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-e2e-mll: learn-order1-e2e-mll.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-full: learn-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-full: predict-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

prune-order1-full: prune-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

forced-align-order1-full: forced-align-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-lat: learn-order1-lat.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-lat: predict-order1-lat.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

prune-random: prune-random.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order2-lat: learn-order2-lat.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order2-lat: predict-order2-lat.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

overlap-vs-per: overlap-vs-per.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-learn: segrnn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-predict: segrnn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-prune: segrnn-prune.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-cascade-learn: segrnn-cascade-learn.o cascade.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-cascade-predict: segrnn-cascade-predict.o cascade.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

