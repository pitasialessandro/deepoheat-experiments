import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from hvp import hvp_fwdfwd
from train import update, get_gpu_memory
from eval import eval_heat3d
from models1 import DeepOHeat_v1, DeepOHeat_phi


# =========================================================
# Baseline loss
# =========================================================
@eqx.filter_jit
def apply_model_deepoheat_st(model, xc, yc, zc, fc, lam_b=1.0):
    def PDE_loss(model, x, y, z, f):
        u = model(((x, y, z), f))

        v_x = jnp.ones(x.shape)
        v_y = jnp.ones(y.shape)
        v_z = jnp.ones(z.shape)

        ux, uxx = hvp_fwdfwd(lambda x_: model(((x_, y, z), f)), (x,), (v_x,), True)
        uy, uyy = hvp_fwdfwd(lambda y_: model(((x, y_, z), f)), (y,), (v_y,), True)
        uz, uzz = hvp_fwdfwd(lambda z_: model(((x, y, z_), f)), (z,), (v_z,), True)

        pde_res = jnp.mean((uxx + uyy + uzz) ** 2)

        bc_top = jnp.mean((uz[:, :, :, -1, :] - f.reshape(-1, 21, 21, 1)) ** 2)
        bc_bottom = jnp.mean((u[:, :, :, 0, :] - 0.2 - 0.2 * uz[:, :, :, 0, :]) ** 2)
        bc_other = (
            jnp.mean((uy[:, :, 0, :, :]) ** 2)
            + jnp.mean((uy[:, :, -1, :, :]) ** 2)
            + jnp.mean((ux[:, 0, :, :, :]) ** 2)
            + jnp.mean((ux[:, -1, :, :, :]) ** 2)
        )

        return pde_res + lam_b * (bc_top + bc_bottom + bc_other)

    loss_fn = lambda model: PDE_loss(model, xc, yc, zc, fc)
    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)
    return loss, gradient


# =========================================================
# Physics-informed loss con due materiali
# =========================================================
@eqx.filter_jit
def apply_model_deepoheat_st_new(
    model,
    xc, yc, zc,
    fc,
    k_region1=1.4,
    k_region2=0.5,
    lam_pde=1.0,
    lam_top=1.0,
    lam_bottom=1.0,
    lam_side=1.0,
    lam_ic=1.0
):
    def loss_fn(model):
        nx = xc.shape[0]
        ny = yc.shape[0]
        nz = zc.shape[0]

        u = model(((xc, yc, zc), fc)).reshape(-1, nx, ny, nz, 1)

        v_x = jnp.ones_like(xc)
        v_y = jnp.ones_like(yc)
        v_z = jnp.ones_like(zc)

        ux, uxx = hvp_fwdfwd(lambda x_: model(((x_, yc, zc), fc)), (xc,), (v_x,), True)
        uy, uyy = hvp_fwdfwd(lambda y_: model(((xc, y_, zc), fc)), (yc,), (v_y,), True)
        uz, uzz = hvp_fwdfwd(lambda z_: model(((xc, yc, z_), fc)), (zc,), (v_z,), True)

        ux  = ux.reshape(-1, nx, ny, nz, 1)
        uxx = uxx.reshape(-1, nx, ny, nz, 1)
        uy  = uy.reshape(-1, nx, ny, nz, 1)
        uyy = uyy.reshape(-1, nx, ny, nz, 1)
        uz  = uz.reshape(-1, nx, ny, nz, 1)
        uzz = uzz.reshape(-1, nx, ny, nz, 1)

        lap = uxx + uyy + uzz
        k_mean = (2.0 * k_region1 * k_region2) / (k_region1 + k_region2)

        # ==========================================
        # PDE residual nei due layer, escludendo bordi e interfaccia
        # ==========================================
        lap_r1 = lap[:, 1:-1, 1:9, 1:-1, :]     # materiale 1
        lap_r2 = lap[:, 1:-1, 10:20, 1:-1, :]   # materiale 2

        res_pde1 = jnp.mean((k_region1 * lap_r1) ** 2)
        res_pde2 = jnp.mean((k_region2 * lap_r2) ** 2)
        loss_pde = res_pde1 + res_pde2

        # ==========================================
        # TOP BC
        # ==========================================
        f_top = fc.reshape(-1, nx, ny, 1)
    
        top1 = jnp.mean(
            (k_region1 * uz[:, :, 1:9, -1, :] - f_top[:, :, 1:9, :]) ** 2
        )
        top_mid = jnp.mean(
            (k_mean * uz[:, :, 9, -1, :] - f_top[:, :, 9, :]) ** 2
        )
        top2 = jnp.mean(
            (k_region2 * uz[:, :, 10:20, -1, :] - f_top[:, :, 10:20, :]) ** 2
        )
        loss_top = top1 + top_mid + top2

        # ==========================================
        # BOTTOM BC
        # ==========================================
        bottom1 = jnp.mean(
            (u[:, :, 1:9, 0, :] - 0.2 - 0.2 * k_region1 * uz[:, :, 1:9, 0, :]) ** 2
        )
        bottom_mid = jnp.mean(
            (u[:, :, 9, 0, :] - 0.2 - 0.2 * k_mean * uz[:, :, 9, 0, :]) ** 2
        )
        bottom2 = jnp.mean(
            (u[:, :, 10:20, 0, :] - 0.2 - 0.2 * k_region2 * uz[:, :, 10:20, 0, :]) ** 2
        )
        loss_bottom = bottom1 + bottom_mid + bottom2

        # ==========================================
        # SIDE BC adiabatiche
        # ==========================================
        loss_side = (
            jnp.mean((k_region1 * uy[:, :, 0, :, :]) ** 2)
            + jnp.mean((k_region2 * uy[:, :, -1, :, :]) ** 2)
            # Nelle pareti che si espandono lungo y (e quindi la derivata direzionale uscente dalle
            # facce e' normale a x) devo calcolare separatamente i tre termini
            + jnp.mean((k_region1 * ux[:, 0, 1:9, :, :]) ** 2)
            + jnp.mean((k_mean * ux[:, 0, 9, :, :]) ** 2)
            + jnp.mean((k_region2 * ux[:, 0, 10:20, :, :]) ** 2)
            + jnp.mean((k_region1 * ux[:, -1, 1:9, :, :]) ** 2)
            + jnp.mean((k_mean * ux[:, -1, 9, :, :]) ** 2)
            + jnp.mean((k_region2 * ux[:, -1, 10:20, :, :]) ** 2)
        )

        # ==========================================
        # Termine di interfaccia
        # ==========================================
        loss_ic = jnp.mean((k_mean * lap[:, 1:-1, 9, 1:-1, :]) ** 2)

        total = (
            lam_pde * loss_pde
            + lam_top * loss_top
            + lam_bottom * loss_bottom
            + lam_side * loss_side
            + lam_ic * loss_ic
        )

        return total

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)
    return loss, gradient


# =========================================================
# Funzione per calcolare separatamente i termini della loss
# =========================================================
@eqx.filter_jit
def compute_loss_terms_deepoheat_st_new(
    model,
    xc, yc, zc,
    fc,
    k_region1=1.4,
    k_region2=0.5,
):
    nx = xc.shape[0]
    ny = yc.shape[0]
    nz = zc.shape[0]

    u = model(((xc, yc, zc), fc)).reshape(-1, nx, ny, nz, 1)

    v_x = jnp.ones_like(xc)
    v_y = jnp.ones_like(yc)
    v_z = jnp.ones_like(zc)

    ux, uxx = hvp_fwdfwd(lambda x_: model(((x_, yc, zc), fc)), (xc,), (v_x,), True)
    uy, uyy = hvp_fwdfwd(lambda y_: model(((xc, y_, zc), fc)), (yc,), (v_y,), True)
    uz, uzz = hvp_fwdfwd(lambda z_: model(((xc, yc, z_), fc)), (zc,), (v_z,), True)

    ux  = ux.reshape(-1, nx, ny, nz, 1)
    uxx = uxx.reshape(-1, nx, ny, nz, 1)
    uy  = uy.reshape(-1, nx, ny, nz, 1)
    uyy = uyy.reshape(-1, nx, ny, nz, 1)
    uz  = uz.reshape(-1, nx, ny, nz, 1)
    uzz = uzz.reshape(-1, nx, ny, nz, 1)

    lap = uxx + uyy + uzz
    k_mean = (2.0 * k_region1 * k_region2) / (k_region1 + k_region2)

    # PDE
    lap_r1 = lap[:, 1:-1, 1:9, 1:-1, :]
    lap_r2 = lap[:, 1:-1, 10:20, 1:-1, :]

    loss_pde = (
        jnp.mean((k_region1 * lap_r1) ** 2)
        + jnp.mean((k_region2 * lap_r2) ** 2)
    )

    # TOP
    f_top = fc.reshape(-1, nx, ny, 1)
    loss_top = (
        jnp.mean((k_region1 * uz[:, :, 1:9, -1, :] - f_top[:, :, 1:9, :]) ** 2)
        + jnp.mean((k_mean * uz[:, :, 9, -1, :] - f_top[:, :, 9, :]) ** 2)
        + jnp.mean((k_region2 * uz[:, :, 10:20, -1, :] - f_top[:, :, 10:20, :]) ** 2)
    )

    # BOTTOM
    loss_bottom = (
        jnp.mean((u[:, :, 1:9, 0, :] - 0.2 - 0.2 * k_region1 * uz[:, :, 1:9, 0, :]) ** 2)
        + jnp.mean((u[:, :, 9, 0, :] - 0.2 - 0.2 * k_mean * uz[:, :, 9, 0, :]) ** 2)
        + jnp.mean((u[:, :, 10:20, 0, :] - 0.2 - 0.2 * k_region2 * uz[:, :, 10:20, 0, :]) ** 2)
    )

    # SIDE
    loss_side = (
        jnp.mean((k_region1 * uy[:, :, 0, :, :]) ** 2)
        + jnp.mean((k_region2 * uy[:, :, -1, :, :]) ** 2)
        + jnp.mean((k_region1 * ux[:, 0, 1:9, :, :]) ** 2)
        + jnp.mean((k_mean * ux[:, 0, 9, :, :]) ** 2)
        + jnp.mean((k_region2 * ux[:, 0, 10:20, :, :]) ** 2)
        + jnp.mean((k_region1 * ux[:, -1, 1:9, :, :]) ** 2)
        + jnp.mean((k_mean * ux[:, -1, 9, :, :]) ** 2)
        + jnp.mean((k_region2 * ux[:, -1, 10:20, :, :]) ** 2)
    )

    # INTERFACCIA
    loss_ic = jnp.mean((k_mean * lap[:, 1:-1, 9, 1:-1, :]) ** 2)

    return loss_pde, loss_top, loss_bottom, loss_side, loss_ic


# =========================================================
# Plot utils
# =========================================================
def plot_loss_terms(history, result_dir):
    epochs = history["epoch"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["loss_pde"], label="loss_pde")
    plt.plot(epochs, history["loss_top"], label="loss_top")
    plt.plot(epochs, history["loss_bottom"], label="loss_bottom")
    plt.plot(epochs, history["loss_side"], label="loss_side")
    plt.plot(epochs, history["loss_ic"], label="loss_ic")
    plt.plot(epochs, history["loss_total"], label="loss_total", linewidth=2, color="black")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Terms During Training")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "loss_terms.png"), dpi=200)
    plt.close()


def plot_loss_terms_separately(history, result_dir):
    terms = ["loss_pde", "loss_top", "loss_bottom", "loss_side", "loss_ic"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, term in enumerate(terms):
        axes[i].plot(history["epoch"], history[term], label=term)
        axes[i].set_title(term)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].set_yscale("log")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    axes[5].plot(history["epoch"], history["loss_total"], label="loss_total", color="black")
    axes[5].set_title("loss_total")
    axes[5].set_xlabel("Epoch")
    axes[5].set_ylabel("Loss")
    axes[5].set_yscale("log")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "loss_terms_separate.png"), dpi=200)
    plt.close()


# =========================================================
# Train generators
# =========================================================
def deepoheat_st_train_generator(fs, batch, nc, key):
    nx = nc
    ny = nc
    nz = (nx // 2) + 1

    idx = jax.random.choice(key, fs.shape[0], (batch,), replace=False)
    fc = fs[idx, :]

    xc = jnp.linspace(0, 1, nx).reshape(-1, 1)
    yc = jnp.linspace(0, 1, ny).reshape(-1, 1)
    zc = jnp.linspace(0, 0.5, nz).reshape(-1, 1)

    return xc, yc, zc, fc


# =========================================================
# Test generator
# =========================================================
def deepoheat_st_test_generator(fs, u):
    x = jnp.linspace(0, 1, 101).reshape(-1, 1)
    y = jnp.linspace(0, 1, 101).reshape(-1, 1)
    z = jnp.linspace(0, 0.5, 51).reshape(-1, 1)
    return x, y, z, fs, u


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training configurations")

    parser.add_argument(
        "--model_name",
        type=str,
        default="DeepOHeat_v1",
        choices=["DeepOHeat_v1", "DeepOHeat_phi"],
    )
    parser.add_argument("--device_name", type=int, default=0, choices=[0, 1])

    # training data
    parser.add_argument("--nc", type=int, default=21)
    parser.add_argument("--batch", type=int, default=50)

    # training settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--log_epoch", type=int, default=100)

    # model settings
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--branch_dim", type=int, default=21**2)
    parser.add_argument("--branch_depth", type=int, default=8)
    parser.add_argument("--branch_hidden", type=int, default=256)
    parser.add_argument("--trunk_depth", type=int, default=3)
    parser.add_argument("--trunk_hidden", type=int, default=64)
    parser.add_argument("--r", type=int, default=128)
    parser.add_argument("--field_dim", type=int, default=1)

    # phi settings
    parser.add_argument("--phi_latent_dim", type=int, default=4)
    parser.add_argument("--y_interface", type=float, default=0.5)

    # fixed conductivity settings
    parser.add_argument("--k_region1", type=float, default=1.4)
    parser.add_argument("--k_region2", type=float, default=0.5)

    # loss weights
    parser.add_argument("--lam_ic", type=float, default=1.0)
    parser.add_argument("--lam_pde", type=float, default=1.0)
    parser.add_argument("--lam_top", type=float, default=1.0)
    parser.add_argument("--lam_bottom", type=float, default=1.0)
    parser.add_argument("--lam_side", type=float, default=1.0)
    parser.add_argument("--lam_data", type=float, default=1.0e-3)

    args = parser.parse_args()

    # =====================================================
    # Load data
    # =====================================================
    fs_train = np.load("data/fs_train_surface_mixed_p4.npy").reshape(-1, 21**2).astype(np.float32)
    fs_test = np.load("data/fs_test_paper_like.npy").reshape(-1, 21**2).astype(np.float32)
    u_test = np.load("data/u_test_custom.npy").astype(np.float32)

    if u_test.ndim == 3:
        u_test = u_test[None, ...]

    fs_train = jnp.asarray(fs_train)
    fs_test = jnp.asarray(fs_test)
    u_test = jnp.asarray(u_test)

    print("fs_train:", fs_train.shape)
    print("fs_test :", fs_test.shape)
    print("u_test  :", u_test.shape)

    # =====================================================
    # Result dir
    # =====================================================
    root_dir = os.path.join(os.getcwd(), "results", "results_surface", args.model_name)

    suffix = (
        f"nf{args.batch}"
        f"_nc{args.nc}"
        f"_branch_{args.branch_depth}_{args.branch_hidden}"
        f"_trunk_{args.trunk_depth}_{args.trunk_hidden}"
        f"_r{args.r}"
    )

    if args.model_name == "DeepOHeat_phi":
        suffix += f"_phi{args.phi_latent_dim}"

    result_dir = os.path.join(root_dir, suffix)
    os.makedirs(result_dir, exist_ok=True)

    # cleanup logs
    for fname in [
        "log (loss).csv",
        "log (eval metrics).csv",
        "total parameters.csv",
        "total runtime (sec).csv",
        "memory usage (mb).csv",
        "loss_terms_history.csv",
    ]:
        fpath = os.path.join(result_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    # header csv loss
    with open(os.path.join(result_dir, "log (loss).csv"), "a") as f:
        f.write("epoch,loss_total,loss_pde,loss_top,loss_bottom,loss_side,loss_ic\n")

    # =====================================================
    # Optimizer
    # =====================================================
    update_fn = update
    schedule = optax.exponential_decay(args.lr, 1000, 0.9)
    optimizer = optax.adam(schedule)

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key, 2)

    # =====================================================
    # Init model
    # =====================================================
    if args.model_name == "DeepOHeat_v1":
        model = DeepOHeat_v1(
            dim=args.dim,
            branch_dim=args.branch_dim,
            field_dim=args.field_dim,
            branch_depth=args.branch_depth,
            branch_hidden=args.branch_hidden,
            trunk_depth=args.trunk_depth,
            trunk_hidden=args.trunk_hidden,
            rank=args.r,
            key=subkey,
        )

    elif args.model_name == "DeepOHeat_phi":
        model = DeepOHeat_phi(
            dim=args.dim,
            branch_dim=args.branch_dim,
            field_dim=args.field_dim,
            branch_depth=args.branch_depth,
            branch_hidden=args.branch_hidden,
            trunk_depth=args.trunk_depth,
            trunk_hidden=args.trunk_hidden,
            rank=args.r,
            key=subkey,
        )

    params = eqx.filter(model, eqx.is_inexact_array)
    num_params = sum(
        jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda x: x.size, params)
        )
    )
    print(f"Total number of parameters: {num_params}")

    opt_state = optimizer.init(params)

    # =====================================================
    # Train/test generator + loss
    # =====================================================
    train_generator = lambda key: deepoheat_st_train_generator(
        fs_train, args.batch, args.nc, key
    )
    test_generator = deepoheat_st_test_generator

    loss_fn = lambda model, xc, yc, zc, fc: apply_model_deepoheat_st_new(
        model,
        xc, yc, zc,
        fc,
        k_region1=args.k_region1,
        k_region2=args.k_region2,
        lam_pde=args.lam_pde,
        lam_top=args.lam_top,
        lam_bottom=args.lam_bottom,
        lam_side=args.lam_side,
        lam_ic=args.lam_ic
    )
    start = None
    
    # =====================================================
    # Training custom con log dei 5 pezzi di loss
    # =====================================================
    history = {
        "epoch": [],
        "loss_total": [],
        "loss_pde": [],
        "loss_top": [],
        "loss_bottom": [],
        "loss_side": [],
        "loss_ic": [],
    }

    for epoch in range(args.epochs):
        key, subkey = jax.random.split(key)
        inputs = train_generator(subkey)

        loss, grads = loss_fn(model, *inputs)
        model, opt_state = update_fn(grads, optimizer, opt_state, model)

        if epoch == 1:
            gpu_memory = get_gpu_memory(args.device_name)
            with open(os.path.join(result_dir, "memory usage (mb).csv"), "a") as f:
                f.write(f"{gpu_memory}\n")
            start = time.time()

        if epoch % args.log_epoch == 0:
            xc, yc, zc, fc = inputs

            loss_pde, loss_top, loss_bottom, loss_side, loss_ic = compute_loss_terms_deepoheat_st_new(
                model,
                xc, yc, zc,
                fc,
                k_region1=args.k_region1,
                k_region2=args.k_region2,
            )

            loss_total_val = (
                args.lam_pde * loss_pde
                + args.lam_top * loss_top
                + args.lam_bottom * loss_bottom
                + args.lam_side * loss_side
                + args.lam_ic * loss_ic
            )

            history["epoch"].append(epoch + 1)
            history["loss_total"].append(float(loss_total_val))
            history["loss_pde"].append(float(loss_pde))
            history["loss_top"].append(float(loss_top))
            history["loss_bottom"].append(float(loss_bottom))
            history["loss_side"].append(float(loss_side))
            history["loss_ic"].append(float(loss_ic))

            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  total  : {float(loss_total_val):.8e}")
            print(f"  pde    : {float(loss_pde):.8e}")
            print(f"  top    : {float(loss_top):.8e}")
            print(f"  bottom : {float(loss_bottom):.8e}")
            print(f"  side   : {float(loss_side):.8e}")
            print(f"  ic     : {float(loss_ic):.8e}")

            with open(os.path.join(result_dir, "log (loss).csv"), "a") as f:
                f.write(
                    f"{epoch+1},"
                    f"{float(loss_total_val)},"
                    f"{float(loss_pde)},"
                    f"{float(loss_top)},"
                    f"{float(loss_bottom)},"
                    f"{float(loss_side)},"
                    f"{float(loss_ic)}\n"
                )

    runtime = time.time() - start if start is not None else 0.0
    
    # =====================================================
    # Save model
    # =====================================================
    eqx.tree_serialise_leaves(
        os.path.join(result_dir, args.model_name + "_trained_model.eqx"),
        model
    )
    
    # =====================================================
    # Save loss history + plots
    # =====================================================
    loss_history_array = np.column_stack([
        np.array(history["epoch"]),
        np.array(history["loss_total"]),
        np.array(history["loss_pde"]),
        np.array(history["loss_top"]),
        np.array(history["loss_bottom"]),
        np.array(history["loss_side"]),
        np.array(history["loss_ic"]),
    ])

    np.savetxt(
        os.path.join(result_dir, "loss_terms_history.csv"),
        loss_history_array,
        delimiter=",",
        header="epoch,loss_total,loss_pde,loss_top,loss_bottom,loss_side,loss_ic",
        comments=""
    )

    plot_loss_terms(history, result_dir)
    plot_loss_terms_separately(history, result_dir)

    
    # =====================================================
    # Eval
    # =====================================================
    (
        rel_l2_mean, rel_l2_std,
        rmse_mean, rmse_std,
        max_l1_mean, max_l1_std,
        mape_mean, mape_std,
        pape_mean, pape_std
    ) = eval_heat3d(
        model,
        test_generator,
        fs_test,
        u_test,
        result_dir,
    )

    print(f"Runtime --> total: {runtime:.2f}sec ({(runtime/(args.epochs-1)*1000):.2f}ms/iter.)")
    print(f"rel_l2 --> mean: {rel_l2_mean:.8f} (std: {rel_l2_std:8f})")
    print(f"rmse   --> mean: {rmse_mean:.8f} (std: {rmse_std:8f})")
    print(f"max_l1 --> mean: {max_l1_mean:.8f} (std: {max_l1_std:8f})")
    print(f"mape   --> mean: {mape_mean:.8f} (std: {mape_std:8f})")
    print(f"pape   --> mean: {pape_mean:.8f} (std: {pape_std:8f})")

    # =====================================================
    # Save stats
    # =====================================================
    np.savetxt(os.path.join(result_dir, "total runtime (sec).csv"), np.array([runtime]), delimiter=",")
    np.savetxt(os.path.join(result_dir, "total parameters.csv"), np.array([num_params]), delimiter=",")

    with open(os.path.join(result_dir, "log (eval metrics).csv"), "a") as f:
        f.write(f"rel_l2_mean: {rel_l2_mean}\n")
        f.write(f"rel_l2_std: {rel_l2_std}\n")
        f.write(f"rmse_mean: {rmse_mean}\n")
        f.write(f"rmse_std: {rmse_std}\n")
        f.write(f"max_l1_mean: {max_l1_mean}\n")
        f.write(f"max_l1_std: {max_l1_std}\n")
        f.write(f"mape_mean: {mape_mean}\n")
        f.write(f"mape_std: {mape_std}\n")
        f.write(f"pape_mean: {pape_mean}\n")
        f.write(f"pape_std: {pape_std}\n")