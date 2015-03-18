import nengo; import numpy as np
def test_ensemble():
    with nengo.Network(seed=1) as model:
        stim = nengo.Node([0.5])
        ens = nengo.Ensemble(40, dimensions=1)
        nengo.Connection(stim, ens)
        probe = nengo.Probe(ens, synapse=0.05)
    sim = nengo.Simulator(model)
    sim.run(0.5)
    assert np.allclose(
        sim.data[probe][sim.trange() > 0.4], 0.5, atol=0.1)
