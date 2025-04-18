import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy

import utils
from logger import Logger
from video import VideoRecorder

from sac_ae import SacAeAgent


'''
python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --work_dir ./log \
    --seed 1
'''

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')

    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step):
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        video.save('%d.mp4' % step)
        # 打印验证模型的平均奖励回报
        L.log('eval/episode_reward', episode_reward, step)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'sac_ae':
        return SacAeAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_latent_lambda=args.decoder_latent_lambda,
            decoder_weight_lambda=args.decoder_weight_lambda,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'), # 是否是像素输入
        height=args.image_size, # 如果是像素输入的话，图像的高度
        width=args.image_size, # 如果是像素输入的话，图像的宽度
        frame_skip=args.action_repeat # 动作重复次数，相当于跳帧
    )
    env.seed(args.seed)

    # stack several consecutive frames together
    # 对于像素输入的情况，使用帧栈来增加时间维度
    # 这里的帧栈是将连续的帧拼接在一起，形成一个新的观测空间
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    # 限制动作空间的每个维度的数值大小都要在[-1, 1]之间
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    # 使用的是重放缓冲区
    # 连续帧存放
    # 离散帧抽取进行训练
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    # 创建agent
    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    # 日志记录器
    L = Logger(args.work_dir, use_tb=args.save_tb)
    
    # episode: 经历了了多少个游戏周期
    # episode_reward: 记录的是一次游戏周期内的总奖励
    # done: 是否结束
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    # 限制了总训练的步数
    for step in range(args.num_train_steps):
        if done:
            # 如果游戏结束则进行验证
            if step > 0:
                # 打印训练的时间以及经过的步数
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                # todo 后续查看这个的功能
                L.dump(step)

            # evaluate agent periodically
            if step % args.eval_freq == 0:
                # 指定的评估频率
                # 打印评估时的步数
                L.log('eval/episode', episode, step)
                evaluate(env, agent, video, args.num_eval_episodes, L, step)
                # 评估后保存模型
                if args.save_model:
                    agent.save(model_dir, step)
                # 可以选择是否保存重放缓冲区，默认不保存
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            # 游戏结束，重置环境
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0 # 游戏周期内经过的步数
            episode += 1

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            # 一开始预热时使用随机采样的动作
            action = env.action_space.sample()
        else:
            # 进入设置为验证模式，退出设置为验证模式
            with utils.eval_mode(agent):
                # 随机动作采样
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            # 如果步数大于预热步数，则进行训练
            # 第一次训练则训练init_steps次，是为了智能体快速利用这些数据进行初步学习
            # 后续每次采样动作后训练一次
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # 对预测的动作进行执行
        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        # 限制游戏周期内的步数
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        # 将采集的步数存入重放缓冲区
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    main()
