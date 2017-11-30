CXXFLAGS += -std=c++11 -I .. -L ../util -L ../nn -L ../autodiff -L ../opt -L ../la -L ../ebt -L ../seg -L ../fst

bin = \
    oracle-error \
    ctc-learn \
    ctc-predict \
    segrnn-learn \
    segrnn-sup-learn \
    segrnn-predict \
    segrnn-align \
    seglin-learn \
    seglin-sup-learn \
    seglin-predict \
    seglin-beam-prune

    # segrnn-loss \
    # ctc-loss \
    # segrnn-prune \
    # segrnn-beam-prune \
    # segrnn-frame-learn \
    # segrnn-ctc-learn \
    # segrnn-sup-loss \
    # seglin-sup-learn \
    # segrnn-forward-learn \
    # segrnn-seg-learn \
    # segrnn-seg-predict \

    # segrnn-cascade-learn \
    # segrnn-cascade-predict \
    # segrnn-hypercolumn-learn \
    # segrnn-entropy-learn \
    # segrnn-input-grad \
    # oracle-cost \
    # learn-order1-full \
    # predict-order1-full \
    # prune-order1-full \
    # learn-order1-lat \
    # predict-order1-lat \
    # forced-align-order1-full \
    # prune-random \
    # lat-order2-learn \
    # lat-order2-predict \
    # overlap-vs-per \

.PHONY: all clean

all: $(bin)

clean:
	-rm *.o
	-rm $(bin)

oracle-error: oracle-error.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

oracle-random: oracle-random.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

oracle-cost: oracle-cost.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

ctc-learn: ctc-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lutil -lnn -lautodiff -lopt -lla -lfst -lebt -lblas

ctc-loss: ctc-loss.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lutil -lnn -lautodiff -lopt -lla -lfst -lebt -lblas

ctc-predict: ctc-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lutil -lnn -lautodiff -lopt -lla -lfst -lebt -lblas

learn-order1-e2e-mll: learn-order1-e2e-mll.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-full: learn-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-full: predict-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

prune-order1-full: prune-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

forced-align-order1-full: forced-align-order1-full.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

learn-order1-lat: learn-order1-lat.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

predict-order1-lat: predict-order1-lat.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

prune-random: prune-random.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

lat-order2-learn: lat-order2-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

lat-order2-predict: lat-order2-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

overlap-vs-per: overlap-vs-per.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-learn: segrnn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-loss: segrnn-loss.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-predict: segrnn-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-forward-learn: segrnn-forward-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-prune: segrnn-prune.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-beam-prune: segrnn-beam-prune.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-align: segrnn-align.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-cascade-learn: segrnn-cascade-learn.o cascade.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-cascade-predict: segrnn-cascade-predict.o cascade.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-input-grad: segrnn-input-grad.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsego -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-entropy-learn: segrnn-entropy-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-hypercolumn-learn: segrnn-hypercolumn-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-frame-learn: segrnn-frame-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-ctc-learn: segrnn-ctc-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-sup-learn: segrnn-sup-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-sup-loss: segrnn-sup-loss.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

seglin-learn: seglin-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

seglin-sup-learn: seglin-sup-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

seglin-predict: seglin-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

seglin-beam-prune: seglin-beam-prune.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-seg-learn: segrnn-seg-learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

segrnn-seg-predict: segrnn-seg-predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lseg -lfst -lutil -lnn -lautodiff -lopt -lla -lebt -lblas

