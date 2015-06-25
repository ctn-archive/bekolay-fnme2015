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
def test_cchannelchain(Simulator, plt, rng, seed, outfile, probes):
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

        if probes:
            p_input = nengo.Probe(stim)
            p_outputs = [nengo.Probe(ens[i], synapse=synapse)
                         for i in range(layers)]

    sim = Simulator(model)
    sim.run(0.5)

    if probes:
        plt.subplot(2, 1, 1)
        plt.plot(sim.trange(), sim.data[p_input], c='k', lw=2)
        for p_output in p_outputs:
            plt.plot(sim.trange(), sim.data[p_output])
        plt.ylabel('Decoded output')
        plt.xticks(())

        plt.subplot(2, 1, 2)
        for p_output in p_outputs:
            plt.plot(sim.trange(), np.sum(np.abs(
                sim.data[p_output] - sim.data[p_input]), axis=1))
        plt.ylabel("Error")
        plt.xlabel('Time (s)')

        with open(outfile, 'a') as outf:
            outf.write('"rmse": %f,\n' % (
                rmse(sim.data[p_outputs[-1]][sim.trange() > 0.4], value)))

    if hasattr(sim, 'close'):
        sim.close()


# --- 2. Product
def test_product(Simulator, plt, seed, outfile, probes):
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

        ens_direct = nengo.Node(output=lambda t, x: x[0] * x[1], size_in=2)
        nengo.Connection(stimulus, ens_direct)
        if probes:
            probe_direct = nengo.Probe(ens_direct)
            probe_test = nengo.Probe(product_net.output, synapse=0.005)

    sim = Simulator(model)
    sim.run(duration + wait_duration)

    if probes:
        after_wait = sim.trange() > wait_duration
        actual = sim.data[probe_test][after_wait]
        target = sim.data[probe_direct][after_wait]

        plt.subplot(2, 1, 1)
        plt.plot(sim.trange()[after_wait], target, c='k', lw=2)
        plt.plot(sim.trange()[after_wait], actual, c='b')
        plt.ylabel('Decoded output (product)')
        plt.xticks(())

        plt.subplot(2, 1, 2)
        plt.plot(sim.trange()[after_wait],
                 np.sum(np.abs(actual - target), axis=1))
        plt.ylabel('Error')
        plt.xlabel('Time (s)')

        with open(outfile, 'a') as outf:
            outf.write('"rmse": %f,\n' % (rmse(actual, target)))

    if hasattr(sim, 'close'):
        sim.close()


# --- 3. Controlled oscillator
#
# This benchmark builds a controlled oscillator and tests it at a sequence of
# inputs. The benchmark computes the FFT of the response of the system and
# compares that to the FFT of perfect sine waves of the desired frequencies.
# The final score is the mean normalized dot product between the FFTs.
def test_controlledoscillator(Simulator, plt, rng, seed, outfile, probes):
    f_max = 2
    T = 2   # time to hold each input for
    stims = np.array([1, 0.5, 0, -0.5, -1])  # control signal
    tau = 0.1

    with nengo.Network(seed=seed) as model:
        state = nengo.Ensemble(n_neurons=500, dimensions=3, radius=1.7)

        def feedback(x):
            x0, x1, f = x
            w = f * f_max * 2 * np.pi
            return x0 + w * tau * x1, x1 - w * tau * x0
        nengo.Connection(state, state[:2], function=feedback, synapse=tau)

        freq = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(freq, state[2], synapse=tau)

        kick = nengo.Node(lambda t: 1 if t < 0.08 else 0)
        nengo.Connection(kick, state[0])

        control = piecewise({i * T: stim for i, stim in enumerate(stims)})
        freq_control = nengo.Node(control)

        nengo.Connection(freq_control, freq)

        if probes:
            p_state = nengo.Probe(state, synapse=0.03)

    sim = Simulator(model)
    sim.run(len(stims) * T)

    if probes:
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

        with open(outfile, 'a') as outf:
            outf.write('"score": %f,\n' % np.mean(score))

        plt.subplot(2, 1, 1)
        lines = plt.plot(np.arange(steps) * dt, data.T)
        plt.xlabel('Time (s)')
        plt.ylabel('Decoded value')
        plt.legend(lines, ['%gHz' % f for f in ideal_freqs],
                   loc='best', prop={'size': 8})

        plt.subplot(2, 1, 2)
        lines = plt.plot(np.fft.fftshift(freqs),
                         np.fft.fftshift(fft, axes=1).T,
                         drawstyle="steps-mid")
        plt.xlim(-f_max * 2, f_max * 2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power of decoded value')
        plt.legend(lines, ['%gHz' % f for f in ideal_freqs],
                   loc='best', prop={'size': 8})

    if hasattr(sim, 'close'):
        sim.close()


# --- 4. SPA sequence with memory
def test_sequencememory(Simulator, plt, seed, outfile, probes):
    dimensions = 512
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
        if probes:
            p_memory = nengo.Probe(model.memory.state.output, synapse=0.1)

    sim = Simulator(model)
    sim.run(items * item_time + mem_time)

    if probes:
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

        with open(outfile, 'a') as outf:
            outf.write('"prob_60000": %f,\n' % prob_cleanup[0])
            outf.write('"prob_1000000000": %f,\n' % prob_cleanup[1])

        lines = plt.plot(trange, dotp)
        plt.legend(lines, ["Pointer %d" % (i + 1) for i in range(items)],
                   loc='best')
        plt.xlabel('Time (s)')
        plt.ylabel('Similarity to memory representation')

    if hasattr(sim, 'close'):
        sim.close()
