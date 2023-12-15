import pytest

from rl4co.envs import PDPEnv, TSPEnv
from rl4co.models import (
    ActiveSearch,
    AttentionModel_NAR,
    AutoregressivePolicy,
    EASEmb,
    EASLay,
    HeterogeneousAttentionModel,
    PPOModel,
    SymNCO,
)
from rl4co.utils import RL4COTrainer


# Test out simple training loop and test with multiple baselines
@pytest.mark.parametrize("baseline", ["rollout", "exponential", "critic", "mean", "no"])
def test_nar_reinforce(baseline):
    env = TSPEnv(num_loc=20)

    model = AttentionModel_NAR(
        env, baseline=baseline, train_data_size=10, val_data_size=10, test_data_size=10
    )

    trainer = RL4COTrainer(max_epochs=1, devices=1)
    trainer.fit(model)
    trainer.test(model)


