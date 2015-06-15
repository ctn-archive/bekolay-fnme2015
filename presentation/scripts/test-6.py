import nengo; import numpy as np
from nengo.tests.conftest import Simulator, plt, seed
from nengo.tests.conftest import pytest_generate_tests

def test_ensemble(Simulator, nl, seed, plt):
    with nengo.Network(seed=seed) as model:
        model.config[nengo.Ensemble].neuron_type = nl()
        stim = nengo.Node([0.5])
        ens = nengo.Ensemble(40, dimensions=1)
        nengo.Connection(stim, ens)
        probe = nengo.Probe(ens, synapse=0.05)
    sim = Simulator(model)
    sim.run(0.5)

    plt.plot(sim.trange(), sim.data[probe])

    assert np.allclose(
        sim.data[probe][sim.trange() > 0.4], 0.5, atol=0.1)
