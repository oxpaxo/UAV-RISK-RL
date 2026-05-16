# ENV V2 Design Report

## Environment

`DynamicObstacleFlowEnv` is implemented in `envs/dynamic_obstacle_flow_env.py` as a separate environment from the frozen legacy 3-ball Gym.

## State And Action

- UAV state uses position, velocity, goal-relative direction, progress, and obstacle-set observations.
- Action is a 3D continuous velocity command, with altitude constrained around a nominal height. This is horizontal avoidance with constrained altitude, not full 3D flight.
- Observation keys remain `ego`, `obs`, `mask`, and `global_risk` for later SB3 compatibility.

## Active Obstacle Flow

- Default active obstacles: 5-8, with `eval_flow_high_density` using 8-10.
- The environment maintains a target active count and replaces removed obstacles during the episode.
- Removal reasons include `passed_by_uav`, `out_of_bounds`, `no_future_threat`, `lifetime`, and `far_from_nominal_path`.

## Motion Models

- `linear`
- `sinusoidal_lateral` with amplitude, period, and phase.
- `accel_decel` with target speed resampling and transition time.
- `ar1_velocity` with phi=0.90 and scenario-dependent noise.
- `crossing_or_sudden_threat` with planned turn time and post-turn crossing toward a planned CPA point.

## Threat Generation

- Each spawned obstacle samples `low`, `medium`, or `high` threat class.
- Planned CPA ranges are high=[0.35,1.15], medium=[1.55,2.35], low=[2.80,4.20].
- Planned TTC is sampled from scenario-specific ranges and is computed against the nominal straight path, independent of the sanity policy trajectory.
- `threat_valid` checks planned CPA class bounds, TTC finiteness, and non-colliding initial placement.

## Train/Eval Scenarios

- `train_flow_mixed`
- `eval_flow_id`
- `eval_flow_high_density`
- `eval_flow_high_speed`
- `eval_flow_high_threat`
- `eval_flow_mixed_ood`
- `eval_flow_sudden_threat`
