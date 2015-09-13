"""A set of accuracy benchmarks.

These mostly come from a set of benchmarks implemented for the
nengo_brainstorms backend, adapted to be run using Nengo test suite.

Contributors:
  - Terry Stewart (CNRG lab)
  - Trevor Bekolay (CNRG lab)
  - Jan Gosmann (CNRG lab)
  - Sam Fok (Brains in Silicon, Stanford University)
  - Alexander Neckar (Brains in Silicon, Stanford University)
  - John Aguayo (Brains in Silicon, Stanford University)
"""
import numpy as np
import seaborn as sns

import nengo
import nengo.utils.builder
import nengo.spa as spa
from nengo.utils.numpy import rmse
from nengo.utils.functions import piecewise

try:
    import nengo_spinnaker
except ImportError:
    pass

from .plots import setup, onecolumn
from .utils import HilbertCurve

colors = sns.color_palette("cubehelix", n_colors=6)


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
        if type(plt).__name__ != 'Mock':
            setup(figsize=(onecolumn * 2, 4.0))
            colors = sns.cubehelix_palette(5)
            lines = []
            for i, p_output in enumerate(p_outputs):
                l = plt.plot(sim.trange(), sim.data[p_output],
                             c=colors[i % len(colors)])
                lines.append(l[0])
            plt.legend(lines, ["Ensemble %d" % i for i in range(1, 6)],
                       loc='best')
            plt.plot(sim.trange(), sim.data[p_input], c='k', lw=1)
            plt.xlim(right=0.12)
            plt.yticks((-0.5, 0, 0.5))
            plt.xticks((0, 0.05, 0.1))
            plt.ylabel('Decoded output')
            plt.xlabel('Time (s)')
            sns.despine()
            plt.saveas = 'results-1.svg'

        outfile.write('"n_neurons": %d,\n' % sum(
            e.n_neurons for e in model.all_ensembles))
        outfile.write('"simtime": 0.5,\n')
        outfile.write('"rmse": %f,\n' % (
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
            probe_inp = nengo.Probe(stimulus, synapse=0.005)
            probe_direct = nengo.Probe(ens_direct)
            probe_test = nengo.Probe(product_net.output, synapse=0.005)

        if 'spinnaker' in Simulator.__module__:
            nengo_spinnaker.add_spinnaker_params(model.config)
            model.config[stimulus].function_of_time = True

    sim = Simulator(model)
    sim.run(duration + wait_duration)

    if probes:
        after_wait = sim.trange() > wait_duration
        actual = sim.data[probe_test][after_wait]
        target = sim.data[probe_direct][after_wait]

        setup(figsize=(onecolumn * 2, 4.0))
        plt.subplot(2, 1, 1)
        y = sim.data[probe_inp][after_wait]
        plt.plot(sim.trange()[after_wait], y[:, 0], c=colors[2])
        plt.plot(sim.trange()[after_wait], y[:, 1], c=colors[3])
        plt.ylabel('Decoded input')
        plt.xlim(left=wait_duration, right=duration + wait_duration)
        plt.xticks(())

        plt.subplot(2, 1, 2)
        plt.plot(sim.trange()[after_wait], actual, c=colors[4])
        plt.plot(sim.trange()[after_wait], target, c='k', lw=1)
        plt.ylabel('Decoded product')
        plt.xlabel('Time (s)')
        plt.xlim(left=wait_duration, right=duration + wait_duration)
        sns.despine()
        plt.saveas = 'results-2.svg'

        outfile.write('"n_neurons": %d,\n' % sum(
            e.n_neurons for e in model.all_ensembles))
        outfile.write('"simtime": %f,\n' % (duration + wait_duration))
        outfile.write('"rmse": %f,\n' % (rmse(actual, target)))

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

        if 'spinnaker' in Simulator.__module__:
            nengo_spinnaker.add_spinnaker_params(model.config)
            model.config[kick].function_of_time = True
            model.config[freq_control].function_of_time = True

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

        outfile.write('"n_neurons": %d,\n' % sum(
            e.n_neurons for e in model.all_ensembles))
        outfile.write('"simtime": %f,\n' % (len(stims) * T))
        outfile.write('"score": %f,\n' % np.mean(score))

        setup(figsize=(onecolumn * 2, 4.0))
        lines = []
        if type(plt).__name__ != 'Mock':
            for i, y in enumerate(np.fft.fftshift(fft, axes=1)):
                lines.append(plt.stem(np.fft.fftshift(freqs), y))
                marker, stem, base = lines[-1]
                marker.set_color(colors[i])
                marker.set_markersize(10.0)
                for s in stem:
                    s.set_color(colors[i])
                    s.set_linewidth(1.0)
                base.set_visible(False)
        plt.xlim(-f_max * 2, f_max * 2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power of decoded value')
        plt.legend(lines, ['%gHz' % f for f in ideal_freqs],
                   loc='best', prop={'size': 8})
        sns.despine()
        plt.saveas = 'results-3.svg'

    if hasattr(sim, 'close'):
        sim.close()


# --- 4. SPA sequence
#
# This builds a basal ganglia that runs through a fixed sequence of actions.
# The value for the benchmark is a measure of the time taken to go from one
# item in the sequence to the next.  We compute both a mean and a standard
# deviation for this timing measure.
def remove_passthrough_nodes(network):
    m = nengo.Network()

    conns = list(network.all_connections)
    inputs, outputs = nengo.utils.builder.find_all_io(conns)

    keep_nodes = []
    for probe in network.all_probes:
        if isinstance(probe.target, nengo.Node):
            if probe.target.output is None:
                keep_nodes.append(probe.target)

    with m:
        for ens in network.all_ensembles:
            m.add(ens)
        for node in network.all_nodes:
            if node.output is None and node not in keep_nodes:
                conns_in = inputs[node]
                conns_out = outputs[node]
                for c in conns_in:
                    conns.remove(c)
                    outputs[c.pre_obj].remove(c)
                for c in conns_out:
                    conns.remove(c)
                    inputs[c.post_obj].remove(c)

                for c_in in conns_in:
                    for c_out in conns_out:
                        c = nengo.utils.builder._create_replacement_connection(
                            c_in, c_out)
                        if c is not None:
                            conns.append(c)
                            outputs[c.pre_obj].append(c)
                            inputs[c.post_obj].append(c)
            else:
                m.add(node)
        for conn in conns:
            m.add(conn)
        for probe in network.all_probes:
            m.add(probe)
    return m


def _test_sequence(Simulator, plt, seed, outfile, probes, prune_passthrough):
    dimensions = 32
    subdimensions = 16
    T = 4.0
    seq_length = 6

    with spa.SPA(seed=seed) as model:
        model.state = spa.Memory(dimensions=dimensions,
                                 subdimensions=subdimensions)

        seq_actions = ['dot(state,A%d) --> state=A%d' % (i, (i+1) % seq_length)
                       for i in range(seq_length)]

        model.bg = spa.BasalGanglia(spa.Actions(*seq_actions))
        model.thal = spa.Thalamus(model.bg)

        def stim_state(t):
            if t < 0.1:
                return 'A0'
            else:
                return '0'

        model.input = spa.Input(state=stim_state)

        if probes:
            p_state = nengo.Probe(model.state.state.output, synapse=0.01)

        if 'spinnaker' in Simulator.__module__:
            nengo_spinnaker.add_spinnaker_params(model.config)
            model.config[
                model.input.input_nodes['state']
            ].function_of_time = True

    vocab = model.get_input_vocab('state')

    if prune_passthrough:
        model = remove_passthrough_nodes(model)

    sim = Simulator(model)
    sim.run(T)

    if probes:
        t = sim.trange()
        data = sim.data[p_state]
        ideal = np.array([vocab.parse('A%d' % i).v for i in range(seq_length)])
        dotp = np.dot(data, ideal.T)

        best = np.argmax(dotp, axis=1)
        delta = np.diff(best)
        indexes = np.where(delta != 0)
        # [:, 1:] ignores the first transition, which is meaningless
        delta_t = np.diff(indexes)[:, 1:] * 0.001

        mean = np.mean(delta_t)
        std = np.std(delta_t)

        outfile.write('"n_neurons": %d,\n' % sum(
            e.n_neurons for e in model.all_ensembles))
        outfile.write('"simtime": %f,\n' % T)
        outfile.write('"timing_mean": %f,\n' % mean)
        outfile.write('"timing_std": %f,\n' % std)

        setup(figsize=(onecolumn * 2, 3.0), palette_args={
            'palette': "cubehelix", 'n_colors': 6})
        plt.plot(t[t < 1.0], dotp[t < 1.0])
        for transition in t[indexes[0]]:
            plt.axvline(transition, c='0.5', lw=1, ls=':')
        plt.ylabel('Similarity to item')
        plt.xlabel('Time (s)')
        plt.xlim(right=1.0)
        sns.despine()
        if prune_passthrough:
            plt.saveas = 'results-4.svg'
        else:
            plt.saveas = 'results-5.svg'

    if hasattr(sim, 'close'):
        sim.close()


def test_sequence(Simulator, plt, seed, outfile, probes):
    _test_sequence(Simulator, plt, seed, outfile, probes,
                   prune_passthrough=False)


def test_sequence_pruned(Simulator, plt, seed, outfile, probes):
    _test_sequence(Simulator, plt, seed, outfile, probes,
                   prune_passthrough=True)
