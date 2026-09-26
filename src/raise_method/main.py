"""Pinned veRL task-runner extension; upstream source remains unchanged."""
import hydra
import ray
from verl.trainer import main_ppo
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.utils.device import auto_set_device


class RaiseTaskRunner(main_ppo.TaskRunner):
    def run(self, config):
        from .trainer import RaiseTrainer
        # The pinned TaskRunner uses this module global as its trainer factory.
        # Scope replacement to this dedicated Ray actor and restore on exit.
        original = main_ppo.RayPPOTrainer
        main_ppo.RayPPOTrainer = RaiseTrainer
        try:
            return super().run(config)
        finally:
            main_ppo.RayPPOTrainer = original


@hydra.main(config_path="configs", config_name="raise", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    shared = config.raise_method.get("shared_trainer", False)
    runner = ray.remote(num_cpus=1)(RaiseTaskRunner) if config.raise_method.method == "raise" or shared else None
    main_ppo.run_ppo(config, task_runner_class=runner)


if __name__ == "__main__":
    main()
