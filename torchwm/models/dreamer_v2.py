import torch
import torch.distributions as distributions

from torchwm.models.dreamer import Dreamer
from torchwm.utils.dreamer_utils import symlog as _symlog


class DreamerV2(Dreamer):
    def _get_head_config(self) -> tuple[str, dict]:
        return "symlog_twohot", {
            "num_buckets": getattr(self.args, "num_buckets", 255),
            "symlog_range": getattr(self.args, "symlog_range", 10.0),
        }

    def _compute_kl_loss(
        self,
        prior: dict,
        posterior: dict,
        post_dist: distributions.Distribution,
        prior_dist: distributions.Distribution,
    ) -> torch.Tensor:
        post_no_grad = self.rssm.detach_state(posterior)
        prior_no_grad = self.rssm.detach_state(prior)
        post_mean_no_grad, post_std_no_grad = (
            post_no_grad["mean"],
            post_no_grad["std"],
        )
        prior_mean_no_grad, prior_std_no_grad = (
            prior_no_grad["mean"],
            prior_no_grad["std"],
        )

        kl_loss = self.args.kl_alpha * torch.mean(
            distributions.kl.kl_divergence(
                self.rssm.get_dist(post_mean_no_grad, post_std_no_grad),
                prior_dist,
            )
        )
        kl_loss += (1 - self.args.kl_alpha) * torch.mean(
            distributions.kl.kl_divergence(
                post_dist,
                self.rssm.get_dist(prior_mean_no_grad, prior_std_no_grad),
            )
        )
        return kl_loss

    def _get_reward_target(self, rews: torch.Tensor) -> torch.Tensor:
        return _symlog(rews[:-1])

    def _compute_actor_loss(
        self, returns: torch.Tensor, discounts: torch.Tensor
    ) -> torch.Tensor:
        weight = discounts.detach()
        target = _symlog(returns.detach())
        return -torch.mean(weight * target)

    def _compute_value_loss(
        self,
        value_feat: torch.Tensor,
        value_targ: torch.Tensor,
        discounts: torch.Tensor,
    ) -> torch.Tensor:
        value_dist = self.value_model(value_feat)
        target = _symlog(value_targ)
        log_prob = value_dist.log_prob(target)
        return -torch.mean(discounts * log_prob)
