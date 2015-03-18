import nengo; import numpy as np
from nengo.tests.conftest import (Simulator, analytics, plt,
    logger, seed, pytest_generate_tests)

def test_ensemble(Simulator, nl, seed, analytics, logger, plt):
    with nengo.Network(seed=seed) as model:
        model.config[nengo.Ensemble].neuron_type = nl()
        stim = nengo.Node([0.5])
        ens = nengo.Ensemble(40, dimensions=1)
        nengo.Connection(stim, ens)
        probe = nengo.Probe(ens, synapse=0.05)
    sim = Simulator(model)
    sim.run(0.5)

    plt.plot(sim.trange(), sim.data[probe])
    analytics.add_data('out', sim.data[probe], "decoded out")
    logger.info('RMSE=%f', nengo.utils.numpy.rmse(
        sim.data[probe][sim.trange() > 0.4], 0.5))

    assert np.allclose(
        sim.data[probe][sim.trange() > 0.4], 0.5, atol=0.1)
