import nengo
def test_ensemble():
    with nengo.Network() as model:
        ens = nengo.Ensemble(40, dimensions=1)
    assert model.ensembles[0] is ens
