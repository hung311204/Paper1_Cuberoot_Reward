from argparse import ArgumentTypeError
import os 

from torch.utils.tensorboard import SummaryWriter

from .imports import * 

seed = 0    # Global variable to store the seed

def set_torch(n_threads: int = 0, deterministic: bool = True, cuda: bool = False) -> th.device:
    th.set_num_threads(n_threads)
    th.backends.cudnn.deterministic = deterministic
    return th.device("cuda" if th.cuda.is_available() and cuda else "cpu")

def set_random_seed(s: int = None):
    global seed
    if s is not None: seed = s
    rnd.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

def str2bool(s: str) -> bool:
    if s.lower() == 'true':return True
    elif s.lower() == 'false': return False
    raise ArgumentTypeError('Boolean value expected.')

def make_logger(run_name, args):
    wandb_path = wb.init(
        name=run_name,
        config=vars(args),
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        sync_tensorboard=True,
    )
    wandb_path = os.path.split(wandb_path.dir)[0]

    '''
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    '''
    #return writer, wandb_path
    return None, wandb_path

def Linear(input_dim: int, output_dim: int, act_fn: str = 'leaky_relu', init_weight_uniform: bool = True) -> nn.Linear:
    """
    Create and initialize a linear layer with appropriate weights.
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/    
    
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        act_fn (str): Activation function.
        init_weight_uniform (bool): Whether to uniformly sample initial weights.

    Returns:
        nn.Linear: The initialized layer.
    """
    act_fn = act_fn.lower()
    gain = nn.init.calculate_gain(act_fn)
    layer = nn.Linear(input_dim, output_dim)
    if init_weight_uniform: nn.init.xavier_uniform_(layer.weight, gain=gain)
    else: nn.init.xavier_normal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.00)
    return layer

class Tanh(nn.Module):
    """
    In-place tanh module.
    """

    def forward(self, input: th.Tensor) -> th.Tensor:
        return th.tanh_(input)

th_act_fns = {
    'tanh': Tanh(),
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU()
}

