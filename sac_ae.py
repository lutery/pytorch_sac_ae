import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder
from decoder import make_decoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf. todo
    函数的主要作用是对动作的输出进行范围限制，将其压缩到 ([-1, 1]) 的范围内，同时对动作的对数概率进行相应的调整。它是基于强化学习中连续动作空间的需求设计的，确保输出的动作值符合环境的动作范围约束
    """
    # 将动作均值限制在[-1, 1]之间
    # 这是因为许多强化学习环境（如 MuJoCo）要求动作值在 ([-1, 1]) 范围内
    mu = torch.tanh(mu)
    if pi is not None:
        # 如果pi不为空，则将pi在[-1, 1]之间
        pi = torch.tanh(pi)
    if log_pi is not None:
        # 在动作经过 tanh 压缩后，其概率分布的对数概率需要重新计算
        # 原因是 tanh 改变了动作的分布形状，导致原始的高斯分布不再适用
        # 调整公式： [ \log \pi(a) = \log \pi(u) - \sum_i \log \left( 1 - \tanh^2(u_i) \right) ] 其中 ( u ) 是未经过 tanh 的动作，( a = \tanh(u) )
        # 这里的 1e-6 是为了避免数值不稳定（如分母为零）
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        '''
        obs_shape: 观测空间的形状 (例如图像的宽、高、通道数)。
        action_shape: 动作空间的形状 (例如动作的维度)。
        hidden_dim: 隐藏层的维度，用于 Actor 网络。
        encoder_type: 编码器的类型 (例如 'pixel' 表示像素输入)。
        encoder_feature_dim: 编码器输出的特征维度。
        log_std_min: Actor 网络中 log_std 的最小值，用于限制动作的标准差。
        log_std_max: Actor 网络中 log_std 的最大值，用于限制动作的标准差。
        num_layers: 编码器和解码器的卷积层数。
        num_filters: 编码器和解码器的卷积核数量。
        '''
        super().__init__()

        # 构建环境编码器
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        '''
        return 动作均值，根据均值和方差采样得到动作，动作对数概率应该是后期用于熵的计算，计算标准差的log值
        '''
        # 提取环境的特征
        obs = self.encoder(obs, detach=detach_encoder)

        # 将特征输入到全连接网络中，得到均值和标准差
        # 从这里来看模型代码应该针对的是连续动作空间
        # 这里预测的是标准差的log值，所以会有负值
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        # 这边应该是限制标准差的范围
        # torch.tanh(log_std)限制在[-1, 1]之间
        log_std = torch.tanh(log_std)
        # 将在[-1, 1]之间的值映射到[log_std_min, log_std_max]之间
        # todo 可能是这个区间的值梯度会更大
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        # 这个事用于记录输出的值
        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        # 对预测的动作进行采样，得到随机值，和离散动作对应随机采样的逻辑一致
        if compute_pi:
            # exp得到标准差
            std = log_std.exp()
            # 得到一个shape和mu一样的噪声
            noise = torch.randn_like(mu)
            # 给预测的均值添加噪音，这里的pi符合均值为mu，标准差为std的正态分布
            # 这里就是对动作进行采样，得到随机的动作
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        # 计算对数概率
        if compute_log_pi:
            # 计算对采样动作的对数概率密度 todo
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function. q值预测"""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        # 结合观测和动作的特征，得到 q 值
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        '''
        obs_shape: 观测空间的形状 (例如图像的宽、高、通道数)。
        action_shape: 动作空间的形状 (例如动作的维度)。
        hidden_dim: 隐藏层的维度，用于 Critic 网络。
        encoder_type: 编码器的类型 (例如 'pixel' 表示像素输入)。
        encoder_feature_dim: 编码器输出的特征维度。
        num_layers: 编码器和解码器的卷积层数。
        num_filters: 编码器和解码器的卷积核数量。
        '''
        super().__init__()


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        # 预测两个Q值
        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class SacAeAgent(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,  # 观测空间的形状 (例如图像的宽、高、通道数)。
        action_shape,  # 动作空间的形状 (例如动作的维度)。
        device,  # 设备 (CPU 或 GPU)，用于模型和张量的计算。
        hidden_dim=256,  # 隐藏层的维度，用于 Actor 和 Critic 网络。
        discount=0.99,  # 折扣因子 γ，用于计算未来奖励的折现值。
        init_temperature=0.01,  # 初始温度参数 α，用于平衡探索和利用。
        alpha_lr=1e-3,  # 温度参数 α 的学习率。
        alpha_beta=0.9,  # 温度参数优化器的动量超参数 β。
        actor_lr=1e-3,  # Actor 网络的学习率。
        actor_beta=0.9,  # Actor 网络优化器的动量超参数 β。
        actor_log_std_min=-10,  # Actor 网络中 log_std 的最小值，用于限制动作的标准差。
        actor_log_std_max=2,  # Actor 网络中 log_std 的最大值，用于限制动作的标准差。
        actor_update_freq=2,  # Actor 网络更新的频率 (每隔多少步更新一次)。
        critic_lr=1e-3,  # Critic 网络的学习率。
        critic_beta=0.9,  # Critic 网络优化器的动量超参数 β。
        critic_tau=0.005,  # 软更新参数 τ，用于更新目标 Critic 网络。
        critic_target_update_freq=2,  # Critic 目标网络更新的频率。
        encoder_type='pixel',  # 编码器的类型 (例如 'pixel' 表示像素输入)。
        encoder_feature_dim=50,  # 编码器输出的特征维度。
        encoder_lr=1e-3,  # 编码器的学习率。
        encoder_tau=0.005,  # 编码器的软更新参数 τ。
        decoder_type='pixel',  # 解码器的类型 (例如 'pixel' 表示像素输入)。
        decoder_lr=1e-3,  # 解码器的学习率。
        decoder_update_freq=1,  # 解码器更新的频率。
        decoder_latent_lambda=0.0,  # 解码器的潜在空间正则化系数。
        decoder_weight_lambda=0.0,  # 解码器权重衰减系数。
        num_layers=4,  # 编码器和解码器的卷积层数。
        num_filters=32  # 编码器和解码器的卷积核数量。
    ):
        self.device = device
        self.discount = discount # 折扣，todo应该是用于q值计算
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda

        # sac算法的动作模型，评价模型，评价目标模型
        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        # 将评价模型的参数复制到目标模型中，可以采用TargetNet
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 同步评价模型的环境特征提取器的参数到动作模型中
        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        # todo 
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        # todo 这个熵的计算数学公式是什么
        self.target_entropy = -np.prod(action_shape)

        # 这里应该就是ae的了，解码器？todo
        self.decoder = None
        if decoder_type != 'identity':
            # create decoder
            # 构建环境解码层
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)

            # todo 单独构建评价模型的环境编码器的优化器
            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            # 单独构建环境特征解码器的优化器
            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

        # optimizers
        # 优化器
        # 整个动作模型的参数优化器（包含编码器）
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        # 整个评价模型的参数优化器（包含编码器）
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        # 构建log_alpha的优化器
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        # 切换训练模型
        self.train()
        # 目标网络也设置为训练模式
        self.critic_target.train()

    def train(self, training=True):
        '''
        设置是否处于训练模式，包含动作模型、评价模型、环境特征解码器，仅限像素特征
        '''
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            # 动作模型根据观测值计算动作
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            # 返回预测的动作
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        # 这里的随机动作采样使用的是基于均值+方差的随机采样的到动作
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        '''
        param obs: 当前观测值
        param action: 当前动作
        param reward: 当前奖励
        param next_obs: 下一个观测值
        param not_done: 是否完成
        param L: 日志记录器
        param step: 当前训练步数
        '''
        with torch.no_grad():
            # 动作网络预测下一个状态的动作采样，下一个状态的动作的对数概率
            _, policy_action, log_pi, _ = self.actor(next_obs)
            # 利用目标评价网络得到预测的下一个状态的Q值
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            # 从预测的Q值中得到最小的Q值
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            # 计算当前状态的Q值
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        # 使用当前的评价网络计算当前状态的Q值
        current_Q1, current_Q2 = self.critic(obs, action)
        # 预测的当前Q值需要和计算出来的目标Q值要接近
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        # 优化critic参数
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 记录日志
        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        # 选择更小的动作Q值
        actor_Q = torch.min(actor_Q1, actor_Q2)
        # - actor_Q 依旧是计算尽可能的小值，使得动作的Q值尽可能的大
        # self.alpha.detach() * log_pi是在计算熵，鼓励探索
        # log_pi：当前策略 ( \pi ) 生成的动作 ( a_t ) 的对数概率，表示策略的熵
        # self.alpha.detach()：温度参数 ( \alpha )，用于平衡熵正则化项的权重。detach() 的作用是防止梯度回传到 ( \alpha )，因为 ( \alpha ) 的更新是独立的
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        # 更新动作网络的参数
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        # 这行代码是在计算 温度参数 ( \alpha ) 的损失函数
        # 作用是平衡策略的熵（随机性）和动作价值（利用性）。
        # self.alpha：当前的温度参数 ( \alpha )，通过 self.log_alpha.exp() 计算得到。
        # log_pi：当前策略生成的动作的对数概率，表示策略的熵
        # self.target_entropy：目标熵值，表示希望策略达到的随机性水平。
        # (-log_pi - self.target_entropy)：表示当前策略的熵与目标熵之间的差距。
        # .detach()：防止梯度回传到 log_pi 和 self.target_entropy，因为它们不需要更新
        # 最终，alpha_loss 是一个标量，表示当前温度参数 ( \alpha ) 的优化目标。
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_decoder(self, obs, target_obs, L, step):
        '''
        更新特征解码器
        obs和target_obs传入的值是相同的

        学习潜在表示：通过重构损失（reconstruction loss），让编码器提取出能够有效表示观测值的低维特征。
        辅助强化学习：编码器的潜在表示被用于策略网络（Actor）和价值网络（Critic）的输入，从而提升学习效率

        为什么需要 AE（自编码器）？
        在 SAC+AE 中，自编码器的主要目的是：

        处理高维观测值：

        在像素级输入（如图像）环境中，直接使用高维观测值进行强化学习效率较低。
        自编码器通过学习低维潜在表示，降低了输入的维度，同时保留了关键信息。
        提升样本效率：

        自编码器的重构损失提供了额外的监督信号，帮助编码器学习更好的特征表示，从而加速策略和价值网络的学习。
        正则化潜在空间：

        通过对潜在表示添加正则化项，鼓励编码器学习到更平滑、更有结构的特征表示。

        这种方法特别适用于像素级输入的强化学习任务。
        '''

        # 采集obs的特征
        h = self.critic.encoder(obs)

        if target_obs.dim() == 4:
            # == 4说明 target_obs 是一个 4D 张量，通常表示图像数据，形状为 (batch_size, channels, height, width)
            # 这种情况下，target_obs 是高维像素数据（如 RGB 图像），需要进行预处理以适配模型的输入要求
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)
        # 对环境进行解码
        rec_obs = self.decoder(h)
        # 解码后的obs要和加了噪声后的目标obs要接近
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        # L2 偏置
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        # 优化环境的特征采样器
        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

        self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update(self, replay_buffer, L, step):
        '''
        params replay_buffer: 经验回放池
        params L: 日志记录器
        params step: 当前训练步数
        '''
        # 进行动作的采样，得到当前的观测值、动作、奖励、下一个观测值和是否完成
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        # 计算本次训练的奖励平均值
        L.log('train/batch_reward', reward.mean(), step)

        # 训练评价网络
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        # 根据指定的训练频率更新动作和alpha
        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        # 根据指定的频率，将评价网络更新到目标评价网络
        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        # 根据指定的频率更新特征解码器，这个应该就是ae部分
        if self.decoder is not None and step % self.decoder_update_freq == 0:
            self.update_decoder(obs, obs, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )
