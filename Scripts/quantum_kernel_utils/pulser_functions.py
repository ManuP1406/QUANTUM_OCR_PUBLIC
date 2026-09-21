"""
Basic pulser functions and pulse definitions.
"""
import numpy as np
from pulser import Pulse, Sequence, Register
from pulser.waveforms import RampWaveform, BlackmanWaveform,ConstantWaveform
from pulser.devices import AnalogDevice, DigitalAnalogDevice
from dataclasses import replace
from pulser.channels.dmm import DMM

Omega_max = 2.0 * 2 * np.pi  # Rabi frequency MAX
U = Omega_max / 2.0

# Radius of interaction between rydberg atoms
try:
    R_interatomic = AnalogDevice.rydberg_blockade_radius(U)
except:
    R_interatomic = 8.0  # Default value if calculation fails
    print("Warning: Could not calculate R_interatomic, using default 8.0")


def create_device_with_dmm():
    """ DMM channel """
    dmm_channel = DMM(
            clock_period=4,
            min_duration=16,
            max_duration=2**26,
            mod_bandwidth=8,
            bottom_detuning=-2 * np.pi * 20,  # detuning between 0 and -20 MHz
            total_bottom_detuning=-2 * np.pi * 2000,  # total detuning
        )
    device = replace(DigitalAnalogDevice.to_virtual(), dmm_objects=(dmm_channel,))
    return device


def compute_detuning_weights(nodes, feature_indices):
    """
    Compute DMM weights as normalized sum of features.
    Returns weights in [0,1] for each node.
    """
    features = np.array([[node[i] for i in feature_indices] for node in nodes])
    sums = features.sum(axis=1)
    min_val, max_val = sums.min(), sums.max()
    if max_val - min_val < 1e-12:
        return np.ones(len(nodes)) * 0.5
    weights = (sums - min_val) / (max_val - min_val)
    return weights.tolist()

def Y_pulse(theta, t_pulse):
    """Create a Y rotation pulse."""
    return Pulse.ConstantDetuning(
        BlackmanWaveform(t_pulse, theta), 0, -np.pi / 2
    )


def Z_pulse(phi, t_pulse):
    """Create a Z rotation pulse."""
    return Pulse.ConstantAmplitude(
        0, BlackmanWaveform(t_pulse, phi), phi
    )


def X_pulse(theta, t_pulse):
    """Create an X rotation pulse."""
    return Pulse.ConstantDetuning(
        BlackmanWaveform(t_pulse, theta), 0, 0
    )


"""
Basic pulser functions and pulse definitions.
"""
import numpy as np
from pulser import Pulse, Sequence, Register
from pulser.waveforms import RampWaveform, BlackmanWaveform, ConstantWaveform
from pulser.devices import AnalogDevice, DigitalAnalogDevice
from dataclasses import replace
from pulser.channels.dmm import DMM

Omega_max = 2.0 * 2 * np.pi
U = Omega_max / 2.0

try:
    R_interatomic = AnalogDevice.rydberg_blockade_radius(U)
except:
    R_interatomic = 8.0
    print("Warning: Could not calculate R_interatomic, using default 8.0")

def create_sequence(nodes, LAYERS, encoding_method, encoding_type,
                   feature_pair=None, feature_idx=None, freq_params=None,
                   use_dmm=False):
    t_pulse_ra = 800
    t_pulse = 400
    detuning = 0.2
    delta0 = 2 * np.pi
    theta_0 = np.pi / 15

    coords = {i: node[:2] for i, node in enumerate(nodes)}
    reg = Register(coords)

    if use_dmm:
        device = create_device_with_dmm()
        seq = Sequence(reg, device)

        seq.declare_channel("ry_glob", "rydberg_global")
        seq.declare_channel("ry_local", "rydberg_local")

        SHIFT = 1.0
        REF = 1.0
        
        feature_indices = [2, 3]
        features = np.array([[node[i] for i in feature_indices] for node in nodes])
        sums = features.sum(axis=1) + SHIFT
        weights = 1.0 - REF / sums
        weights = np.clip(weights, 0, 1)
        dmm_weights = weights.tolist()

        qubit_ids = list(reg.qubit_ids)
        weight_dict = {qid: w for qid, w in zip(qubit_ids, dmm_weights)}
        detuning_map = reg.define_detuning_map(weight_dict)
        seq.config_detuning_map(detuning_map, "dmm_0")

        t_list = seq.declare_variable("t_list", size=LAYERS)
        theta_list = seq.declare_variable("theta_list", size=LAYERS)

        seq.add(Y_pulse(theta_0, t_pulse), "ry_glob")

        for l in range(LAYERS):
            detuning_waveform = ConstantWaveform(t_list[l], -delta0)
            seq.add_dmm_detuning(detuning_waveform, "dmm_0")
            constant_pulse = Pulse.ConstantAmplitude(
                Omega_max, RampWaveform(t_list[l], 0, 0), 0.0
            )
            seq.add(constant_pulse, "ry_glob")
            seq.add(Y_pulse(theta_list[l], t_pulse), "ry_glob")
            

        

    else:
        seq = Sequence(reg, DigitalAnalogDevice)
        seq.declare_channel("ry_glob", "rydberg_global")
        seq.declare_channel("ry_local", "rydberg_local")

        if encoding_type == "with_hyperfine" and encoding_method in ["multi_run", "single_run"]:
            seq.declare_channel("ra_local", "raman_local")

            if encoding_method == "multi_run":
                fi, fj = feature_pair

                for i in range(len(nodes)):
                    seq.target(i, "ra_local")
                    param_i = freq_params[0][fi - 2]
                    param_j = freq_params[0][fj - 2]
                    scaled_i = nodes[i][2] * param_i
                    scaled_j = nodes[i][3] * param_j

                    if scaled_i < 0 or scaled_j < 0:
                        raise ValueError(f"Negative pulse amplitude detected!")
                    seq.add(Y_pulse(scaled_i, t_pulse_ra), "ra_local")
                    seq.add(X_pulse(scaled_j, t_pulse_ra), "ra_local")

                t_list = seq.declare_variable("t_list", size=LAYERS)
                theta_list = seq.declare_variable("theta_list", size=LAYERS)

                for l in range(LAYERS):
                    for i in range(len(nodes)):
                        seq.target(i, "ra_local")
                        param_i = freq_params[l][fi - 2]
                        param_j = freq_params[l][fj - 2]
                        scaled_i = nodes[i][2] * param_i
                        scaled_j = nodes[i][3] * param_j

                    constant_pulse = Pulse.ConstantAmplitude(
                        Omega_max, RampWaveform(t_list[l], detuning, detuning), 0.0
                    )
                    seq.add(constant_pulse, "ry_glob")

            elif encoding_method == "single_run":
                for i in range(len(nodes)):
                    seq.target(i, "ra_local")
                    param_f = freq_params[0][feature_idx]
                    scaled_f = nodes[i][2] * param_f
                    seq.add(Y_pulse(scaled_f, t_pulse_ra), "ra_local")

                seq.add(X_pulse(np.pi, t_pulse_ra), "ry_glob")

                t_list = seq.declare_variable("t_list", size=LAYERS)
                theta_list = seq.declare_variable("theta_list", size=LAYERS)

                for l in range(LAYERS):
                    for i in range(len(nodes)):
                        seq.target(i, "ra_local")
                        param_f = freq_params[l][feature_idx]
                        scaled_f = nodes[i][2] * param_f
                        seq.add(Y_pulse(scaled_f, t_pulse_ra), "ra_local")

                    constant_pulse = Pulse.ConstantAmplitude(
                        Omega_max, RampWaveform(t_list[l], detuning, detuning), 0.0
                    )
                    seq.add(constant_pulse, "ry_glob")
                    seq.add(Y_pulse(theta_list[l], t_pulse), "ry_glob")

        else:
            t_list = seq.declare_variable("t_list", size=LAYERS)
            theta_list = seq.declare_variable("theta_list", size=LAYERS)

            seq.add(Y_pulse(theta_0, t_pulse), "ry_glob")

            for t, theta in zip(t_list, theta_list):
                constant_pulse = Pulse.ConstantAmplitude(
                    Omega_max, RampWaveform(t, detuning, detuning), 0.0
                )
                seq.add(constant_pulse, "ry_glob")
                seq.add(Y_pulse(theta, t_pulse), "ry_glob")

    return seq
