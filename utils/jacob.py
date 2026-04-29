# jrngc_minimal.py
import warnings, re
# 屏蔽 weight_norm 的弃用提示
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*weight_norm.*is deprecated.*"
)
# 屏蔽 torch.cuda.amp.GradScaler 的弃用提示
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.GradScaler.*deprecated.*"
)
# 屏蔽 torch.cuda.amp.autocast 的弃用提示
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.autocast.*deprecated.*"
)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp
import pdb

# ---------------------------
# 1) 残差积木（保留你给的实现）
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, input, hidden, output, dropout):
        super().__init__()
        self.linear_1 = torch.nn.utils.weight_norm(nn.Linear(input, hidden))
        self.linear_2 = nn.Linear(hidden, output)
        self.linear_res = nn.Linear(input, output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(output)

    def forward(self, x):
        # x: [Batch, hidden]
        h = self.linear_1(x)
        h = self.relu(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        res = self.linear_res(x)
        out = h + res
        out = self.layernorm(out)
        return out

    def struct_loss(self):
        return torch.sum(self.linear_res.weight ** 2)


# ------------------------------------------
# 2) Jacobian F-范数正则（Hutchinson 近似）
#    ||J||_F^2 = E_v ||∂(v·y)/∂x||^2,  v~N(0,I)
# ------------------------------------------
class JacobianReg(nn.Module):
    def __init__(self, n: int = 1):
        super().__init__()
        self.n = int(n)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, lag]  (requires_grad=True)
        y: [B, D]
        返回 0.5 * E_v ||∂(v·y)/∂x||_F^2 的近似（与论文一致，常用 n=1）
        """
        B, D = y.shape
        total = 0.0
        for _ in range(max(1, self.n)):
            v = torch.randn(B, D, device=y.device, dtype=y.dtype)
            s = (y * v).sum()  # 标量
            (grad_x,) = torch.autograd.grad(
                s, x, create_graph=True, retain_graph=True, allow_unused=False
            )
            total = total + (grad_x.pow(2).sum() / B)
        # 0.5 系数与论文公式对应；如你不需要可去掉
        return 0.5 * total / max(1, self.n)


# ---------------------------
# 3) 模型主体（保留你给的实现）
# ---------------------------
class JRNGC(nn.Module):
    def __init__(self, d, lag, layers, hidden, dropout,
                 jacobian_lam=1e-3, struct_loss_choice='JF', JFn=1, relu=False):
        super().__init__()
        self.d = d
        self.lag = lag
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.jacobian_lam = jacobian_lam
        self.struct_loss_choice = struct_loss_choice  # 'JF' or 'JL1' or 'none'
        self.JFn = JFn
        self.relu = relu

        self.inputgate = nn.Linear(d * lag, hidden)
        self.outputgate = nn.Linear(hidden, d)
        self.inputgate = torch.nn.utils.weight_norm(self.inputgate)
        self.encoders = nn.ModuleList(
            [ResidualBlock(hidden, hidden, hidden, dropout) for _ in range(layers)]
        )
        self.JReg = JacobianReg(n=JFn)


    def forward(self, x):
        """
        x: [batch, d, T=lag]  →  y: [batch, d]
        """
        x = x.flatten(start_dim=1).to(torch.float32)
        x = self.inputgate(x)
        if self.relu:
            x = F.relu(x)
        for net in self.encoders:
            x = net(x)
        x = self.outputgate(x)
        return x

    def jacobian_causal(self, x, flag=False):
        """
        x: [batch, d, T=lag]
        返回: [d_out, d_in, lag] 的 |J| 在 batch 上的均值
        """
        # eval() 只会关掉 dropout/BN，不会关梯度；我们仍显式开启梯度
        if not flag:
            self.eval()

        # 确保在可求导的图里
        x = x.to(next(self.parameters()).device)
        # 如果外层处于 no_grad 环境，这里强制开启
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            B, d, L = x.shape
            # 在输入的 device 上分配，梯度累积用 float32 更稳
            jac = torch.zeros((d, d, L), device=x.device, dtype=torch.float32)

            # 不用 no_grad；autocast 可关掉以避免半精度梯度问题（CPU 上尤需）
            # 如果你确定在 CUDA 并且想开混合精度，可以把 enabled=True
            with amp.autocast(enabled=False):
                for j in range(d):
                    y = self(x)[:, j]                      # [B]
                    # 更稳的写法：用 autograd.grad 而不是 backward + x.grad
                    (grad_x,) = torch.autograd.grad(
                        y, x, grad_outputs=torch.ones_like(y),
                        retain_graph=True, create_graph=False
                    )                                       # [B, d, L]
                    jac[j] = grad_x.abs().mean(dim=0)       # [d, L] 平均掉 batch

        # 现在 jac 形状是 [d_out=j, d_in=i, lag]
        return jac


    def jacobian_causal_train(self, x):
        """
        训练期用于 JL1 的雅可比（保留计算图），返回 [d_out, d_in, lag]
        x: [batch, d, T=lag]
        """
        x.requires_grad_(True)
        B, d, L = x.shape
        jac = torch.zeros((B, d, d, L), device=x.device, dtype=torch.float32)
        with amp.autocast():
            for j in range(d):
                y = self(x)[:, j]                            # [B]
                (grad_x,) = torch.autograd.grad(
                    y, x, create_graph=True,
                    grad_outputs=torch.ones_like(y)
                )
                jac[:, j, :, :] = grad_x
        jac = jac.abs().mean(dim=0)                          # [d, d, L]
        return jac

    # --- 正则：JF(F-范数) / JL1 ---
    def compute_jacobian_F_loss(self, x):
        if x.ndim == 2: x = x.unsqueeze(0)
        x.requires_grad_(True)
        y = self(x)
        return self.jacobian_lam * self.JReg(x, y)

    def jacobian_causal_L1_loss(self, x):
        if x.ndim == 2: x = x.unsqueeze(0)
        jac = self.jacobian_causal_train(x)                  # [d,d,lag]
        return self.jacobian_lam * jac.sum()

    # --- 经验损失：MSE(历史 → 当前) ---
    def exper_loss(self, x_hist_with_target):
        """
        x_hist_with_target: [batch, d, T=lag+1]; 前 lag 帧为输入，最后 1 帧为标签
        """
        return self.loss_fn(self(x_hist_with_target[:, :, :-1]),
                            x_hist_with_target[:, :, -1])

# -------------------------------------------------
# 4) 数据切片 + 训练/验证划分 + 早停（沿用你的逻辑）
# -------------------------------------------------
def _windows_from_TD(x_TD: np.ndarray, lag: int, device) -> torch.Tensor:
    """
    x_TD: [T, D] 的 numpy 数组
    返回：x_win [N, D, lag+1]，每条样本 = 过去 lag 帧 + 当前 1 帧
    """
    # pdb.set_trace()

    T, D = x_TD.shape
    x = torch.tensor(x_TD, device=device, dtype=torch.float32)    # [T, D]
    x = x.T.unsqueeze(0)                                          # [1, D, T]
    xw = x.transpose(1, 2).unfold(1, lag + 1, 1)                  # [1, N, D, lag+1]
    xw = xw.reshape(-1, xw.shape[2], xw.shape[3])                 # [N, D, lag+1]
    return xw

def _split_train_val(xw: torch.Tensor, val_ratio: float = 0.2):
    """
    时间上连续划分，避免信息泄漏
    xw: [N, D, lag+1]
    """
    N = xw.shape[0]
    n_val = max(1, int(N * val_ratio))
    n_tr = N - n_val
    return xw[:n_tr], xw[n_tr:]

def train_jrngc_on_array(
    x_TD: np.ndarray,              # 你的数据: [T, D]
    lag: int = 10,
    hidden: int = 128,
    layers: int = 2,
    dropout: float = 0.1,
    jacobian_lam: float = 1e-3,
    struct_loss_choice: str = 'JF',# 'JF' / 'JL1' / 'none'
    JFn: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = None,
    val_ratio: float = 0.2,
    min_iter: int = 1000,
    max_iter: int = 10000,
    lookback: int = 10,
    check_first: int = 50,
    check_every: int = 100,
    verbose: bool = True,
):
    """
    训练模型（MSE + Jacobian 正则），带验证早停（沿用你的逻辑）
    返回: model, (训练曲线字典)
    """
    torch.manual_seed(seed); np.random.seed(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # pdb.set_trace()
    # 切成滑动窗口
    xw = _windows_from_TD(x_TD, lag, device)          # [N, D, lag+1]
    x_tr, x_va = _split_train_val(xw, val_ratio)

    D = x_tr.shape[1]
    model = JRNGC(D, lag, layers, hidden, dropout, jacobian_lam,
                  struct_loss_choice, JFn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler()

    best_it = None
    best_loss = torch.inf
    logs = {"it": [], "train_loss": [], "train_pred_loss": [],
            "val_pred_loss": []}


    for it in range(max_iter):
        model.train()
        with amp.autocast():
            pred_loss = model.exper_loss(x_tr)                # MSE
            if struct_loss_choice == 'JL1':
                struct_loss = model.jacobian_causal_L1_loss(x_tr[:, :, :-1])
            elif struct_loss_choice == 'JF':
                struct_loss = model.compute_jacobian_F_loss(x_tr[:, :, :-1])
            else:
                struct_loss = 0.0
            loss = pred_loss + struct_loss

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        # 按原逻辑的检查频率与早停
        if (it < min_iter and (it + 1) % check_first == 0) or ((it + 1) % check_every == 0):
            with torch.no_grad():
                model.eval()
                va_pred = model.exper_loss(x_va) / D
            mean_pred = (pred_loss / D).detach().item()
            mean_loss = (loss / D).detach().item()

            logs["it"].append(it)
            logs["train_pred_loss"].append(mean_pred)
            logs["train_loss"].append(mean_loss)
            logs["val_pred_loss"].append(va_pred.detach().item())

            if verbose:
                print(f"[it {it:6d}] train_pred={mean_pred:.4e}  "
                      f"train_loss={mean_loss:.4e}  val_pred={va_pred:.4e}  "
                      f"best={best_loss:.4e}")

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
            elif (it - best_it) >= lookback * check_every and it > min_iter:
                if verbose:
                    print("Early stopping.")
                break

    return model, logs


# -------------------------------------------------
# 5) 读出因果：full-time J 与 summary GC
# -------------------------------------------------
@torch.no_grad()
def infer_fulltime_and_summary(model: JRNGC, x_TD: np.ndarray, lag: int,
                               summary_mode: str = "max", device: str = None):
    """
    返回：
      J_abs: [D, D, lag]  （full-time 因果强度）
      GC:    [D, D]       （summary，默认对 lag 取 max）
      Lag*:  [D, D]       （主导滞后，1..lag；对角线为 0）
    """
    if device is None:
        device = next(model.parameters()).device
    xw = _windows_from_TD(x_TD, lag, device)          # [N, D, lag+1]
    x_in = xw[:, :, :-1]                              # [N, D, lag]
    J = model.jacobian_causal(x_in).cpu().numpy()     # [D, D, lag]

    # summary 聚合
    if summary_mode == "max":
        GC = J.max(axis=2)
    elif summary_mode == "l1":
        GC = J.sum(axis=2)
    elif summary_mode == "l2":
        GC = np.sqrt((J**2).sum(axis=2))
    else:
        raise ValueError("summary_mode ∈ {'max','l1','l2'}")

    # 主导滞后
    Lag_star = J.argmax(axis=2) + 1                   # 1..lag
    np.fill_diagonal(GC, 0.0)
    np.fill_diagonal(Lag_star, 0)
    return J, GC, Lag_star

def top_k_percent_binarize(GC, percent=0.3):
    """
    提取 GC 中前 percent 比例最大的边，返回二值矩阵（0,1）。
    """
    GC_flat = GC.flatten()
    # 排除对角线的0值再排序
    GC_flat_nonzero = GC_flat[GC_flat > 0]
    threshold = np.percentile(GC_flat_nonzero, 100 * (1 - percent))
    GC_binary = (GC >= threshold).astype(int)
    np.fill_diagonal(GC_binary, 0)  # 确保对角线为0
    return GC_binary
# -------------------------------------------------
# 6) 一个最小用法示例
# -------------------------------------------------
if __name__ == "__main__":
    # 假设你的数据 X: [T=1000, D=116]
    T, D = 200, 116
    X = np.random.randn(T, D).astype(np.float32)

    model, logs = train_jrngc_on_array(
        X, lag=10, hidden=128, layers=2, dropout=0.1,
        jacobian_lam=1e-3, struct_loss_choice='JF', JFn=1,
        lr=1e-3, seed=0, val_ratio=0.2, verbose=True,
        min_iter=500, max_iter=5000, lookback=10, check_first=50, check_every=100
    )

    J, GC, Lag_star = infer_fulltime_and_summary(model, X, 10, summary_mode="max")
    GC_binary = top_k_percent_binarize(GC, percent=0.3)
    print(GC_binary)
