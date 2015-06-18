"""A set of accuracy benchmarks.

These mostly come from a set of benchmarks implemented for the
nengo_brainstorms backend, adapted to be run using Nengo test suite.

Contributors:
  - Terry Stewart (CNRG lab)
  - Trevor Bekolay (CNRG lab)
  - Sam Fok (Brains in Silicon, Stanford University)
  - Alexander Neckar (Brains in Silicon, Stanford University)
  - John Aguayo (Brains in Silicon, Stanford University)
"""
import numpy as np

import nengo
import nengo.spa as spa
from nengo.utils.numpy import rmse
from nengo.utils.functions import piecewise

from .utils import HilbertCurve


# --- 1. Communication channel chain
def test_cchannelchain(Simulator, plt, rng, seed, outfile):
    dims = 2
    layers = 5
    n_neurons = 100
    synapse = nengo.Lowpass(0.01)

    with nengo.Network(seed=seed) as model:
        value = nengo.dists.UniformHypersphere().sample(
            dims, 1, rng=rng)[:, 0]
        stim = nengo.Node(value)

        ens = [nengo.Ensemble(n_neurons, dimensions=dims)
               for _ in range(layers)]

        nengo.Connection(stim, ens[0])
        for i in range(layers - 1):
            nengo.Connection(ens[i], ens[i+1], synapse=synapse)

        p_input = nengo.Probe(stim)
        p_output = nengo.Probe(ens[-1], synapse=synapse)

    sim = Simulator(model)
    sim.run(1.0)

    plt.plot(sim.trange(), sim.data[p_input], c='k')
    plt.plot(sim.trange(), sim.data[p_output], c='b')

    with open(outfile, 'w') as outf:
        outf.write('{\n"rmse": %f,\n' % (
            rmse(sim.data[p_output][sim.trange() > 0.8], value)))

    if hasattr(sim, 'close'):
        sim.close()


# --- 2. Product
def test_product(Simulator, plt, seed, outfile):
    hc = HilbertCurve(n=4)
    duration = 5.
    wait_duration = 0.5

    def stimulus_fn(t):
        return np.squeeze(hc(t / duration).T * 2 - 1)

    model = nengo.Network(seed=seed)
    with model:
        stimulus = nengo.Node(
            output=lambda t: stimulus_fn(max(0., t - wait_duration)),
            size_out=2)

        product_net = nengo.networks.Product(100, 1)
        nengo.Connection(stimulus[0], product_net.A)
        nengo.Connection(stimulus[1], product_net.B)
        probe_test = nengo.Probe(product_net.output)

        ens_direct = nengo.Node(output=lambda t, x: x, size_in=2)
        result_direct = nengo.Node(size_in=1)
        nengo.Connection(stimulus, ens_direct)
        nengo.Connection(
            ens_direct, result_direct, function=lambda x: x[0] * x[1],
            synapse=None)
        probe_direct = nengo.Probe(result_direct)

    sim = Simulator(model)
    sim.run(duration + wait_duration)

    selection = sim.trange() > wait_duration
    test = sim.data[probe_test][selection]
    direct = sim.data[probe_direct][selection]

    plt.plot(sim.trange(), sim.data[probe_direct], c='k')
    plt.plot(sim.trange(), sim.data[probe_test], c='b')

    with open(outfile, 'w') as outf:
        outf.write('{\n"rmse": %f,\n' % (rmse(test, direct)))

    if hasattr(sim, 'close'):
        sim.close()


# --- 3. Controlled oscillator
#
# This benchmark builds a controlled oscillator and tests it at a sequence of
# inputs. The benchmark computes the FFT of the response of the system and
# compares that to the FFT of perfect sine waves of the desired frequencies.
# The final score is the mean normalized dot product between the FFTs.
def test_controlledoscillator(Simulator, plt, rng, seed, outfile):
    f_max = 2
    T = 2   # time to hold each input for
    stims = np.array([1, 0.5, 0, -0.5, -1])   # control signal
    synapse = 0.1

    with nengo.Network(seed=seed) as model:
        state = nengo.Ensemble(n_neurons=500, dimensions=3, radius=1.7)

        def feedback(x):
            x0, x1, f = x
            w = f * f_max * 2 * np.pi
            return x0 + w * synapse * x1, x1 - w * synapse * x0
        nengo.Connection(state, state[:2], function=feedback, synapse=synapse)

        freq = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(freq, state[2], synapse=synapse)

        stim = nengo.Node(lambda t: 1 if t < 0.08 else 0)
        nengo.Connection(stim, state[0])

        control = piecewise({i * T: stims[i] for i in range(len(stims))})
        freq_control = nengo.Node(control)

        nengo.Connection(freq_control, freq)

        p_state = nengo.Probe(state, synapse=0.03)

    sim = Simulator(model)
    sim.run(len(stims) * T)

    data = sim.data[p_state][:, 1]

    ideal_freqs = f_max * stims  # target frequencies

    dt = 0.001
    steps = int(T / dt)
    freqs = np.fft.fftfreq(steps, d=dt)

    # compute fft for each input
    data.shape = len(stims), steps
    fft = np.fft.fft(data, axis=1)

    # compute ideal fft for each input
    ideal_data = np.zeros_like(data)
    for i in range(len(stims)):
        ideal_data[i] = np.cos(2 * np.pi
                               * ideal_freqs[i]
                               * np.arange(steps) * dt)
    ideal_fft = np.fft.fft(ideal_data, axis=1)

    # only consider the magnitude
    fft = np.abs(fft)
    ideal_fft = np.abs(ideal_fft)

    # compute the normalized dot product between the actual and ideal ffts
    score = np.zeros(len(stims))
    for i in range(len(stims)):
        score[i] = np.dot(fft[i] / np.linalg.norm(fft[i]),
                          ideal_fft[i] / np.linalg.norm(ideal_fft[i]))

    with open(outfile, 'w') as outf:
        outf.write('{\n"score": %f,\n' % np.mean(score))

    plt.subplot(2, 1, 1)
    lines = plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(fft, axes=1).T)
    plt.xlim(-f_max * 2, f_max * 2)
    plt.xlabel('FFT of decoded value (Hz)')
    plt.title('Score: %1.4f' % np.mean(score))
    plt.legend(lines, ['%1.3f' % s for s in score],
               loc='best', prop={'size': 8})

    plt.subplot(2, 1, 2)
    lines = plt.plot(np.arange(steps) * dt, data.T)
    plt.xlabel('decoded value')
    plt.legend(lines, ['%gHz' % f for f in ideal_freqs],
               loc='best', prop={'size': 8})

    if hasattr(sim, 'close'):
        sim.close()


# --- 4. SPA sequence with memory
def test_sequencememory(Simulator, plt, seed, outfile):
    dimensions = 256
    subdimensions = 16
    item_time = 0.15
    mem_time = 3.0
    input_strength = 0.1
    items = 4

    with spa.SPA(seed=seed) as model:
        model.memory = spa.Memory(dimensions=dimensions,
                                  subdimensions=subdimensions)

        def memory(t):
            index = int(t / item_time)
            if index < items:
                return '%g*A%d' % (input_strength, index)
            else:
                return '0'

        model.input = spa.Input(memory=memory)
        p_memory = nengo.Probe(model.memory.state.output, synapse=0.1)

    sim = Simulator(model)
    sim.run(items * item_time + mem_time)

    vocab = model.get_input_vocab('memory')
    memory = sim.data[p_memory]

    trange = sim.trange()
    correct = np.array([vocab.parse('A%d' % i).v for i in range(items)]).T

    dotp = np.dot(memory, correct)

    # measure accuracy at two different vocabulary sizes
    #  60000: number of basic terms for an adult human
    #  1000000000: guesstimate of number of complex bound terms (~60000**2)
    Ms = [60000, 1000000000]

    score = np.zeros((len(Ms), items))

    for j, M in enumerate(Ms):
        for i in range(items):
            score[j, i] = vocab.prob_cleanup(dotp[-1, i], M)

    prob_cleanup = np.mean(score, axis=1)

    with open(outfile, 'w') as outf:
        outf.write('{\n"prob_60000": %f,\n' % prob_cleanup[0])
        outf.write('"prob_1000000000": %f,\n' % prob_cleanup[1])

    plt.plot(trange, dotp)

    if hasattr(sim, 'close'):
        sim.close()
