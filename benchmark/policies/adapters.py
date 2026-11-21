"""Load policy factories and create independent vehicle sessions."""
from importlib import import_module
from typing import Protocol


def configure_inference(config, seed):
    """Seed supplied Torch methods without requiring Torch for other plugins."""
    use_torch = config.get('seed_torch', config.get('kind') in
                           ('lead', 'uniad'))
    if not use_torch:
        return dict(seed=seed, framework='adapter_managed', bitwise_repeatability_guaranteed=False)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if 'allow_tf32' in config:
        torch.backends.cuda.matmul.allow_tf32 = config['allow_tf32']
        torch.backends.cudnn.allow_tf32 = config['allow_tf32']
    return dict(seed=seed, framework='torch', cudnn_benchmark=False,
                cudnn_deterministic=True,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                bitwise_repeatability_guaranteed=False)


class VehicleSession(Protocol):
    def reset(self, plan): ...
    def step(self, inputs, speed, position_xyz, yaw_deg, *, gps_position_xyz=None): ...
    def close(self): ...


def policy_factory(entry):
    module, name = entry.split(':')
    factory = getattr(import_module(module), name)
    if not callable(factory):
        raise TypeError(f'Policy factory is not callable: {entry}')
    return factory


def load_policy(config):
    kind=config.get('kind')
    if config.get('factory'):
        owner=policy_factory(config['factory'])(config)
        for attribute in ('sensors','description','new_vehicle','close'):
            if not hasattr(owner,attribute):raise TypeError('Policy owner needs '+attribute)
        return owner
    if kind=='lead':
        from benchmark.policies.lead import LEADPolicy
        return LEADPolicy(config['source_root'],config['model_dir'],config.get('device','cuda:0'),
            config.get('navigation_sample_factor_m',200.),config.get('checkpoint'),
            config.get('controller_variant','official'),config.get('longitudinal_mode'))
    raise ValueError('No adapter registered for ' + str(kind))


def close_policy(policy):
    if policy is not None and callable(getattr(policy,'close',None)):policy.close()
