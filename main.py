import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from models import ConstantVelocity, CoordinatedTurn
from filter import LinearKalmanBehavior
from filter import ExtendedKalmanBehavior, UnscentedendKalmanBehavior
from filter import KalmanFilter, IMMFilter


def parse_arguments():
    """Parse optional arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfg/sim1.yaml',
                        help='configuration file.')
    parser.add_argument('--seed', type=int, default=-1,
                        help='seed of random variables.')
    parser.add_argument('--output_dir', default='result',
                        help='output directory.')
    return parser.parse_args()


def make_imm_filter(cfg):
    """Make an IMM filter."""

    to_model_cls = {
        'ConstantVelocity': ConstantVelocity,
        'CoordinatedTurn': CoordinatedTurn,
    }

    to_behavior_cls = {
        'LKF': LinearKalmanBehavior,
        'EKF': ExtendedKalmanBehavior,
        'UKF': UnscentedendKalmanBehavior,
    }

    kfs = []
    dt = cfg['sampling_period']
    param_models = cfg['dynamics']['model']
    param_kfs = cfg['imm_filter']['kalman_filter']

    for prm_kf in param_kfs:

        idx = prm_kf['model_idx']
        prm_model = param_models[idx]

        model_cls = to_model_cls[prm_model['kind']]
        model = model_cls(dt=dt)

        covar_w = np.diag(np.array(prm_model['sigma_w'])**2)
        covar_v = np.diag(np.array(prm_model['sigma_v'])**2)
        kf_behavior_cls = to_behavior_cls[prm_kf['kind']]
        kf_behavior = kf_behavior_cls(model, covar_w, covar_v)

        kf = KalmanFilter(kf_behavior)
        kf.init_posterior(P=np.diag(prm_kf['initial_P']))
        kfs.append(kf)

    mode_proba = np.array(cfg['imm_filter']['mode_proba'])
    transition_mat = np.array(cfg['imm_filter']['transition_mat'])
    imm_filter = IMMFilter(kfs, mode_proba, transition_mat)

    return imm_filter


def extract_models(cfg):
    """Extract models and parameters."""

    to_model_cls = {
        'ConstantVelocity': ConstantVelocity,
        'CoordinatedTurn': CoordinatedTurn,
    }

    models, covar_ws, covar_vs, initial_xs = [], [], [], []

    for param in cfg['dynamics']['model']:

        cls = to_model_cls[param['kind']]
        models.append(cls(dt=cfg['sampling_period']))

        covar_ws.append(np.diag(np.array(param['sigma_w'])**2))
        covar_vs.append(np.diag(np.array(param['sigma_v'])**2))
        initial_xs.append(np.array(param['initial_x']))

    return models, covar_ws, covar_vs, initial_xs


def generate_time_series_data(cfg):
    """Generate time series data."""

    def _generate_noise(covar):
        mean = np.zeros(covar.shape[0])
        return np.random.multivariate_normal(mean, covar)

    models, covar_ws, covar_vs, initial_xs = extract_models(cfg)

    steps = cfg['steps']
    change_interval = cfg['dynamics']['change_interval']

    dim_x = max([m.NDIM['x'] for m in models])
    dim_z = max([m.NDIM['z'] for m in models])

    x_hist = np.zeros((steps, dim_x))
    z_hist = np.zeros((steps, dim_z))
    mode_hist = np.zeros(steps, dtype=np.int32)

    mode = cfg['dynamics']['initial_model_idx']
    initial_x = np.array(initial_xs[mode])
    x_hist[0, :initial_x.shape[0]] = initial_x
    model, covar_w, covar_v = models[mode], covar_ws[mode], covar_vs[mode]

    for t in range(steps):

        # change model
        if (t > 0) and (t % change_interval == 0):
            mode = (mode + 1) % len(models)
            model = models[mode]
            covar_w, covar_v = covar_ws[mode], covar_vs[mode]

        w = _generate_noise(covar_w)
        v = _generate_noise(covar_v)

        ndim_x, ndim_z = model.NDIM['x'], model.NDIM['z']

        if t > 0:
            x_hist[t, :ndim_x] = model.state_equation(
                                    t-1, x_hist[t-1, :ndim_x], 0, w)

        z_hist[t, :ndim_z] = model.observation_equation(
                                    t, x_hist[t, :ndim_x], v)
        mode_hist[t] = mode

    return x_hist, z_hist, mode_hist


def estimate_state(steps, imm_filter, z_real):
    """Estimate states using an IMM filter and measurements. """

    x_est = np.zeros((steps, imm_filter.ndim_aug))
    mode_proba = np.zeros((steps, len(imm_filter.kalman_filters)))
    x_est[0, 0], x_est[0, 2] = z_real[0, 0], z_real[0, 1]

    imm_filter.init_posterior(x=x_est[0, :])

    for t in range(1, steps):
        x_est[t], _ = imm_filter.estimate(t, z_real[t], 0)
        mode_proba[t] = imm_filter.mode_proba

    return x_est, mode_proba


def plot_result(args, x_est, mode_proba, x_real, z_real, mode_real):
    """Plot a estimation result."""

    os.makedirs(args.output_dir, exist_ok=True)

    plt.figure(1, (8, 7))
    plt.scatter(z_real[:, 0], z_real[:, 1], s=15, c='#377eb8')
    plt.plot(x_real[:, 0], x_real[:, 2], 'yellowgreen')
    plt.plot(x_est[:, 0], x_est[:, 2], 'r')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')
    plt.legend(['measurment', 'state (truth)', 'state (estimate)'])
    plt.grid(True)
    fname = os.path.join(args.output_dir, f'trajectory')
    if args.seed >= 0:
        fname += f'_seed{args.seed}'
    plt.savefig(fname + '.png')

    steps = x_est.shape[0]
    plt.figure(2, (8, 11))
    plt.subplot(311)
    plt.plot(range(steps), np.array(mode_real))
    plt.plot(range(steps), mode_proba[:, 1])
    plt.legend(['mode (truth)', 'mode (estimate)'], loc='upper left')
    plt.ylabel('Mode')
    plt.grid(True)

    plt.subplot(312)
    plt.plot(range(steps), np.array(x_real[:, 1])*3600/1000)
    plt.plot(range(steps), np.array(x_est[:, 1])*3600/1000)
    plt.legend(['truth', 'estimate'], loc='upper left')
    plt.legend(['vx (truth)', 'vx (estimate)'], loc='lower left')
    plt.ylabel('Velocity-x [km/h]')
    plt.grid(True)

    plt.subplot(313)
    plt.plot(range(steps), np.array(x_real[:, 3])*3600/1000)
    plt.plot(range(steps), np.array(x_est[:, 3])*3600/1000)
    plt.legend(['vy (truth)', 'vy (estimate)'], loc='lower left')
    plt.xlabel('Steps')
    plt.ylabel('Velocity-y [km/h]')
    plt.grid(True)
    plt.subplots_adjust(bottom=0.1, top=0.95)
    fname = os.path.join(args.output_dir, f'mode-velocity')
    if args.seed >= 0:
        fname += f'_seed{args.seed}'
    plt.savefig(fname + '.png')


def main():

    args = parse_arguments()

    if args.seed >= 0:
        np.random.seed(args.seed)

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    imm_filter = make_imm_filter(cfg)
    x_real, z_real, mode_real = generate_time_series_data(cfg)
    x_est, mode_proba = estimate_state(cfg['steps'], imm_filter, z_real)

    plot_result(args, x_est, mode_proba, x_real, z_real, mode_real)


if __name__ == '__main__':
    main()
