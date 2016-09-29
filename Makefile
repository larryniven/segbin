CXXFLAGS += -std=c++11 -I .. -L ../speech -L ../nn -L ../autodiff -L ../opt -L ../la -L ../ebt -L ../seg

bin = \
    learn-order1 \
    predict-order1 \
    prune-order1 \
    learn-latent-order1 \
    learn-latent-order1-e2e \
    learn-order1-e2e \
    predict-order1-e2e \
    learn-order1-segnn \
    predict-order1-segnn \
    learn-order1-e2e-ff \
    learn-latent-order1-e2e-ff \
    predict-order1-e2e-ff \
    learn-fw-order1 \
    fw-duality-gap \
    oracle-error \
    oracle-cost \
    learn-ctc \
    predict-ctc \
    learn-order1-full \
    predict-order1-full \
    prune-order1-full \
    loss-order1-full \
    learn-order1-lat \
    predict-order1-lat \
    forced-align-order1-full \
    prune-random

.PHONY: all clean

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

learn-order1: learn-order1.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1: predict-order1.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

prune-order1: prune-order1.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-latent-order1: learn-latent-order1.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-fw-order1: learn-fw-order1.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

fw-duality-gap: fw-duality-gap.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order2: learn-order2.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-e2e: learn-order1-e2e.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-latent-order1-e2e: learn-latent-order1-e2e.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-e2e: predict-order1-e2e.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-segnn: learn-order1-segnn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-segnn: predict-order1-segnn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-e2e-ff: learn-order1-e2e-ff.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-latent-order1-e2e-ff: learn-latent-order1-e2e-ff.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-e2e-ff: predict-order1-e2e-ff.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-latent-order1-e2e-lstm2d: learn-latent-order1-e2e-lstm2d.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-e2e-lstm2d: predict-order1-e2e-lstm2d.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

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

loss-order1-full: loss-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

forced-align-order1-full: forced-align-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-lat: learn-order1-lat.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-lat: predict-order1-lat.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

prune-random: prune-random.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lspeech -lnn -lautodiff -lopt -lla -lebt -lblas

