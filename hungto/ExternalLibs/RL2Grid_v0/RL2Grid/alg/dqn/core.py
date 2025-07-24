from time import time

from stable_baselines3.common.buffers import ReplayBuffer

from .agent import QNetwork
from .config import get_alg_args
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.imports import ap, F, np, optim, th
from hungto.ExternalLibs.RL2Grid_v0.RL2Grid.common.logger import Logger

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class DQN:        
    def __init__(self, envs, max_steps, run_name, start_time, args, ckpt):        
        if not ckpt.resumed: args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        #assert args.n_envs == 1, "Vectorized envs are not supported at the moment"

        track = args.track
        if track: logger = Logger(run_name, args)

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        qnet = QNetwork(envs, args).to(device)
        if ckpt.resumed: qnet.load_state_dict(ckpt.loaded_run['qnet'])

        qnet_optim = optim.Adam(qnet.parameters(), lr=args.lr)
        if ckpt.resumed: qnet_optim.load_state_dict(ckpt.loaded_run['qnet_optim'])
        
        tg_qnet = QNetwork(envs, args).to(device)
        tg_qnet.load_state_dict(qnet.state_dict())

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

        for step in range(init_step, int(args.total_timesteps // args.n_envs)):
            global_step += args.n_envs
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(
                args.eps_start, args.eps_end, args.eps_decay_frac * args.total_timesteps, global_step
            )
            if np.random.rand() < epsilon:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with th.no_grad():
                    actions = qnet.get_action(th.tensor(obs).to(device))
                    
            # TRY NOT TO MODIFY: execute the game and log data.
            try:
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            except:
                ckpt.set_record(args, qnet, global_step, qnet_optim, logger.wb_path, step)
                ckpt.save()
                #envs.close()
                if track: logger.close()

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos['final_info']:
                    if info and "episode" in info:
                        survival = info['episode']['l'][0]/max_steps
                        if args.verbose: print(f"global_step={global_step}, length={info['episode']['l'][0]}, survival={survival*100:.3f}%, return={info['episode']['r'][0]}")
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
                if global_step % args.train_freq == 0:
                    data = rb.sample(args.batch_size)
                    with th.no_grad():
                        tg_act = qnet(data.next_observations).argmax(dim=1, keepdim=True)
                        tg_mac = tg_qnet(data.next_observations).gather(1, tg_act).squeeze()
                        td_target = data.rewards.flatten() + args.gamma * tg_mac * (1 - data.dones.flatten())

                    old_val = qnet(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    # optimize the model
                    qnet_optim.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(qnet.parameters(), 10)
                    qnet_optim.step()
                    
                    if global_step % 100 == 0:
                        steps_per_second =  int(global_step / (time() - start_time))
                        if args.verbose: print(f"SPS={steps_per_second}")
                    '''
                        if track:
                            writer.add_scalar("losses/td_loss", loss, global_step)
                            writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    '''
                # update target network
                if global_step % args.tg_qnet_freq == 0:
                    for tg_qnet_param, qnet_param in zip(tg_qnet.parameters(), qnet.parameters()):
                        tg_qnet_param.data.copy_(
                            args.tau * qnet_param.data + (1.0 - args.tau) * tg_qnet_param.data
                        )

            # If we reach the node's time limit, we just exit the training loop, save metrics and checkpoint
            if (time() - start_time) / 60 >= args.time_limit:
                break

        ckpt.set_record(args, qnet, global_step, qnet_optim, logger.wb_path, step)
        ckpt.save()
        envs.close()
        if track: logger.close()
          