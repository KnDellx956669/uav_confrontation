import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

import math


class SimpleAttention(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.W_query = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.W_key = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.W_val = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, k, q, v, mask=None):
        Q = torch.matmul(q, self.W_query)
        K = torch.matmul(k, self.W_key)
        V = torch.matmul(v, self.W_val)

        norm_factor = 1 / math.sqrt(Q.shape[-1])
        compat = norm_factor * torch.matmul(Q, K.transpose(-1, -2))
        if mask is not None: compat[mask.bool()] = -math.inf
        # 为了在这里解决 0*nan = nan 的问题，输入必须将V中的nan转化为0
        score = torch.nan_to_num(F.softmax(compat, dim=-1), 0)
        return torch.matmul(score, V)


class RMAPPOPolicy:
    """
    This is another version of RMAPPOPolicy using hmp2g for reference
    Mention:
    This is not MAPPO, IPPO at some point
    """
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space  # obs_space.shape = (24, )
        self.share_obs_space = cent_obs_space  # No need for this
        self.act_space = act_space

        self._use_feature_normalization = args.use_feature_normalization
        self.n_action = act_space.shape[0]
        self.hidden_size = args.hidden_size
        self.obs_dim = self.obs_space.shape[-1]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(self.obs_dim)

        # # # # # # # # # # # # Actor-Critic Share # # # # # # # # # # # #
        self.obs_encoder = nn.Sequential(nn.Linear(self.obs_dim, self.hidden_size), nn.ReLU(inplace=True), nn.Linear(self.hidden_size, self.hidden_size))
        self.attention_layer = SimpleAttention(h_dim=self.hidden_size)

        # # # # # # # # # # # #        Actor          # # # # # # # # # # # #
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size // 2, self.n_action))

        # # # # # # # # # # # #        Critic          # # # # # # # # # # # #
        self.ct_encoder = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(inplace=True), nn.Linear(self.hidden_size, self.hidden_size))
        self.ct_attention_layer = SimpleAttention(h_dim=self.hidden_size)
        self.get_value = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(inplace=True), nn.Linear(self.hidden_size, 1))

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        if self._use_feature_normalization:
            obs = self.feature_norm(obs)

        # # # # # # # # # # actor-critic share # # # # # # # # # # # #
        baec = self.obs_encoder(obs)
        baec = self.attention_layer(k=baec, q=baec, v=baec, mask=masks)

        # # # # # # # # # # actor # # # # # # # # # # # #
        logits = self.policy_head(baec)

        # choose action selector
        logit2act = self._logit2act

        # apply action selector
        actions, action_log_probs, distEntropy, probs = logit2act(logits, avail_act=available_actions)

        # # # # # # # # # # critic # # # # # # # # # # # #
        ct_bac = self.ct_encoder(baec)
        ct_bac = self.ct_attention_layer(k=ct_bac, q=ct_bac, v=ct_bac)
        values = self.get_value(ct_bac)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        cent_obs is no use and no difference with local obs here, but save it for the purpose of alignment with MAPPO
        original edition
        :param cent_obs:
        :param rnn_states_critic:
        :param masks:
        :return:
        """
        if self._use_feature_normalization:
            obs = self.feature_norm(cent_obs)

        # # # # # # # # # # actor-critic share # # # # # # # # # # # #
        baec = self.obs_encoder(cent_obs)
        baec = self.attention_layer(k=baec, q=baec, v=baec, mask=masks)

        # # # # # # # # # # critic # # # # # # # # # # # #
        ct_bac = self.ct_encoder(baec)
        ct_bac = self.ct_attention_layer(k=ct_bac, q=ct_bac, v=ct_bac)
        values = self.get_value(ct_bac)

        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        if self._use_feature_normalization:
            obs = self.feature_norm(obs)

        # # # # # # # # # # actor-critic share # # # # # # # # # # # #
        baec = self.obs_encoder(obs)
        baec = self.attention_layer(k=baec, q=baec, v=baec, mask=masks)

        # # # # # # # # # # actor # # # # # # # # # # # #
        logits = self.policy_head(baec)

        # choose action selector
        logit2act = self._logit2act

        # apply action selector
        actions, action_log_probs, dist_entropy, probs = logit2act(logits, avail_act=available_actions)

        # # # # # # # # # # critic # # # # # # # # # # # #
        ct_bac = self.ct_encoder(baec)
        ct_bac = self.ct_attention_layer(k=ct_bac, q=ct_bac, v=ct_bac)
        values = self.get_value(ct_bac)

        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Most used in eval mode or render mode, temporarily no use
        :param obs:
        :param rnn_states_actor:
        :param masks:
        :param available_actions:
        :param deterministic:
        :return:
        """
        if self._use_feature_normalization:
            obs = self.feature_norm(obs)

        # # # # # # # # # # actor-critic share # # # # # # # # # # # #
        baec = self.obs_encoder(obs)
        baec = self.attention_layer(k=baec, q=baec, v=baec, mask=masks)

        # # # # # # # # # # actor # # # # # # # # # # # #
        logits = self.policy_head(baec)

        # choose action selector
        logit2act = self._logit2act

        # apply action selector
        actions, action_log_probs, dist_entropy, probs = logit2act(logits, avail_act=available_actions)

        return actions, rnn_states_actor

    def _logit2act(self, logits_agent_cluster, eval_mode=False, test_mode=False, eval_actions=None, available_actions=None, **kwargs):
        if available_actions is not None: logits_agent_cluster = torch.where(available_actions > 0, logits_agent_cluster)
        act_dist = Categorical(logits=logits_agent_cluster)
        if not test_mode:
            act = act_dist.sample() if not eval_mode else eval_actions
        else:
            act = torch.argmax(act_dist.probs, dim=2)
        actLogProbs = self._get_act_log_probs(act_dist, act)  # the policy gradient loss will feedback from here
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs

    @staticmethod
    def _get_act_log_probs(distribution, action):
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)
