from time import time

from stable_baselines3.common.buffers import ReplayBuffer

from .agent import Actor, QNetwork
from .config import get_alg_args
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.imports import ap, F, np, optim, th
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.logger import Logger

class TD3:
    def __init__(self, envs, max_steps, run_name, start_time, args, ckpt):     
        if not ckpt.resumed: args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        track = args.track
        if track: logger = Logger(run_name, args)
        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        actor = Actor(envs, args).to(device)
        critic1 = QNetwork(envs, args).to(device)
        critic2 = QNetwork(envs, args).to(device)

        if ckpt.resumed:
            actor.load_state_dict(ckpt.loaded_run['actor'])
            critic1.load_state_dict(ckpt.loaded_run['critic'])
            critic2.load_state_dict(ckpt.loaded_run['critic2'])
 
        tg_actor = Actor(envs, args).to(device)
        tg_actor.load_state_dict(actor.state_dict())
        tg_critic1 = QNetwork(envs, args).to(device)
        tg_critic2 = QNetwork(envs, args).to(device)
        tg_critic1.load_state_dict(critic1.state_dict())
        tg_critic2.load_state_dict(critic2.state_dict())

        actor_optim = optim.Adam(list(actor.parameters()), lr=args.actor_lr)
        critic_optim = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=args.critic_lr)
        if ckpt.resumed:
            actor_optim.load_state_dict(ckpt.loaded_run['actor_optim'])
            critic_optim.load_state_dict(ckpt.loaded_run['critic_optim'])

        envs.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=args.n_envs,
            handle_timeout_termination=False
        )

        # TRY NOT TO MODIFY: start the game
        init_step = 1 if not ckpt.resumed else ckpt.loaded_run['last_step']
        global_step = 0 if not ckpt.resumed else ckpt.loaded_run['global_step']
        start_time = start_time
        obs, _ = envs.reset(seed=args.seed)

        for step in range(init_step, args.total_timesteps):
            global_step += args.n_envs
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with th.no_grad():
                    actions = actor(th.tensor(obs).to(device))
                    actions += th.normal(0, actor.action_scale * args.exploration_noise)
                    actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

            # TRY NOT TO MODIFY: execute the game and log data.
            try:
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            except:
                ckpt.set_record(args, actor, critic1, critic2, global_step, actor_optim, critic_optim, logger.wb_path, step)
                ckpt.save()
                #envs.close()
                if track: logger.close()
                
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos['final_info']:
                    if info and "episode" in info:
                        survival = info['episode']['l'][0]/max_steps
                        if args.verbose: print(f"global steps={global_step}, length={info['episode']['l'][0]}, survival={survival*100:.3f}%, return={info['episode']['r'][0]}")
                        if track: logger.store_metrics(survival, info)
                        break

            if track and global_step % logger.log_freq == 0: logger.log_metrics(global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                with th.no_grad():
                    clipped_noise = (th.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                        -args.noise_clip, args.noise_clip
                    ) * tg_actor.action_scale

                    next_state_actions = (tg_actor(data.next_observations) + clipped_noise).clamp(
                        envs.single_action_space.low[0], envs.single_action_space.high[0]
                    )
                    critic1_next_tg = tg_critic1(data.next_observations, next_state_actions)
                    critic2_next_tg = tg_critic2(data.next_observations, next_state_actions)
                    min_qf_next_tg = th.min(critic1_next_tg, critic2_next_tg)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_tg).view(-1)

                critic1_a_values = critic1(data.observations, data.actions).view(-1)
                critic2_a_values = critic2(data.observations, data.actions).view(-1)
                critic1_loss = F.mse_loss(critic1_a_values, next_q_value)
                critic2_loss = F.mse_loss(critic2_a_values, next_q_value)
                critic_loss = critic1_loss + critic2_loss

                # optimize the model
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                if global_step % args.actor_train_freq == 0:
                    actor_loss = -critic1(data.observations, actor(data.observations)).mean()
                    actor_optim.zero_grad()
                    actor_loss.backward()
                    actor_optim.step()

                    # update the tg network
                    for param, tg_param in zip(actor.parameters(), tg_actor.parameters()):
                        tg_param.data.copy_(args.tau * param.data + (1 - args.tau) * tg_param.data)
                    for param, tg_param in zip(critic1.parameters(), tg_critic1.parameters()):
                        tg_param.data.copy_(args.tau * param.data + (1 - args.tau) * tg_param.data)
                    for param, tg_param in zip(critic2.parameters(), tg_critic2.parameters()):
                        tg_param.data.copy_(args.tau * param.data + (1 - args.tau) * tg_param.data)

                '''
                if global_step % 100 == 0:
                    steps_per_second =  int(global_step / (time() - start_time))
                    if args.verbose: print(f"global_step={global_step}, SPS={steps_per_second}")
                    if track:
                        writer.add_scalar("losses/critic1_values", critic1_a_values.mean().item(), global_step)
                        writer.add_scalar("losses/critic2_values", critic2_a_values.mean().item(), global_step)
                        writer.add_scalar("losses/critic1_loss", critic1_loss.item(), global_step)
                        writer.add_scalar("losses/critic2_loss", critic2_loss.item(), global_step)
                        writer.add_scalar("losses/critic_loss", critic_loss.item() / 2.0, global_step)
                        writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                '''
            # If we reach the node's time limit, we just exit the training loop, save metrics and checkpoint
            if (time() - start_time) / 60 >= args.time_limit:
                break

        ckpt.set_record(args, actor, critic1, critic2, global_step, actor_optim, critic_optim, logger.wb_path, step)
        ckpt.save()
        envs.close()
        if track: logger.close()
