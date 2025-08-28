import os
from functools import partial
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from cppn import CPPN, FlattenCPPNParameters
import util

def hue_circ_dist(a, b):
    # hue distance on the circle [0,1)
    return jnp.abs(jnp.remainder(a - b + 0.5, 1.0) - 0.5)

def mirror_coords(xyd):  # xyd: [...,3] with [:,0]=x, [:,1]=y, [:,2]=d
    x, y, d = xyd[...,0:1], xyd[...,1:2], xyd[...,2:3]
    return jnp.concatenate([-x, y, d], axis=-1)


parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")

group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("model")
group.add_argument("--arch", type=str, default="", help="architecture")
group = parser.add_argument_group("data")
group.add_argument("--img_file", type=str, default=None, help="path of image file")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=100000, help="number of iterations")
group.add_argument("--lr", type=float, default=3e-3, help="learning rate")
group.add_argument("--init_scale", type=str, default="default", help="initialization scale")

group.add_argument("--lambda_sym", type=float, default=0.0,
                   help="weight for mirror-consistency regularizer")


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    """
    Train a CPPN on a given image using SGD.
    Specify the architecture and img_file to train on.
    """
    print(args)

    target_img = jnp.array(plt.imread(args.img_file)[:, :, :3])

    cppn = FlattenCPPNParameters(CPPN(args.arch, init_scale=args.init_scale))
    # cppn = FlattenCPPNParameters(CPPN(args.arch))

    rng = jax.random.PRNGKey(args.seed)
    params = cppn.init(rng)

   # def loss_fn(params, target_img):
    #    img = cppn.generate_image(params, img_size=256)
     #   return jnp.mean((img - target_img)**2)

    def loss_fn(params, target_img):
    # Generate current image from CPPN (values already in [0,1])
        img = cppn.generate_image(params, img_size=256)  # [H, W, 3]

    # Baseline reconstruction loss
        mse = jnp.mean((img - target_img) ** 2)

    # Symmetry-consistency regularizer (horizontal flip), optional
        if args.lambda_sym > 0.0:
            sym = jnp.mean(jnp.abs(img - img[:, ::-1, :]))
            return mse + args.lambda_sym * sym

        return mse


    @jax.jit
    def train_step(state, _):
        loss, grad = jax.value_and_grad(loss_fn)(state.params, target_img)
        grad = grad / jnp.linalg.norm(grad)
        state = state.apply_gradients(grads=grad)
        return state, loss

    tx = optax.adam(learning_rate=args.lr)
    state = TrainState.create(apply_fn=None, params=params, tx=tx)

    gen_img_fn = jax.jit(partial(cppn.generate_image, img_size=256))
    losses, imgs_train = [], [gen_img_fn(state.params)]
    pbar = tqdm(range(args.n_iters//100))
    for i_iter in pbar:
        state, loss = jax.lax.scan(train_step, state, None, length=100)
        # state, (loss, grad_norm) = jax.lax.scan(train_step, state, None, length=1)
        # print(loss, grad_norm)
        losses.append(loss)

        pbar.set_postfix(loss=loss.mean().item())
        if i_iter < 100:
            img = gen_img_fn(state.params)
            imgs_train.append(img)

    losses = np.array(jnp.concatenate(losses))
    imgs_train = np.array(jnp.stack(imgs_train))
    params = state.params
    img = gen_img_fn(params)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        util.save_pkl(args.save_dir, "args", args)
        util.save_pkl(args.save_dir, "arch", args.arch)
        util.save_pkl(args.save_dir, "params", params)
        plt.imsave(f"{args.save_dir}/img.png", np.array(img))

        util.save_pkl(args.save_dir, "losses", losses)
        # util.save_pkl(args.save_dir, "imgs_train", imgs_train)

if __name__ == '__main__':
    main(parse_args())



