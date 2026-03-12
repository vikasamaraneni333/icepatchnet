import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """
    A configurable patch-based CNN for Sea Ice Extent prediction.

    Parameters
    ----------
    num_layers : int, optional
        Number of convolutional layers. Default is 3.
    kernel_size : int or list of int, optional
        Kernel size for each layer.
        - If a single int  → same kernel used for ALL layers.
        - If a list of int → one kernel size per layer (length must match num_layers).
        Default is 3 (i.e., 3×3 kernels on all layers).
    base_filters : int, optional
        Number of filters in the first hidden layer. Each subsequent hidden
        layer doubles the filters. Default is 64.

    Examples
    --------
    # Default (matches original model — 3 layers, 3×3 kernels)
    model = CNNModel()

    # 5 layers, all with 3×3 kernels
    model = CNNModel(num_layers=5)

    # 4 layers, all with 5×5 kernels
    model = CNNModel(num_layers=4, kernel_size=5)

    # 4 layers, each with a different kernel size
    model = CNNModel(num_layers=4, kernel_size=[3, 5, 5, 3])
    """

    def __init__(self, num_layers=3, kernel_size=3, base_filters=64):

        super(CNNModel, self).__init__()

        # ── Validate num_layers ──────────────────────────────────────
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"`num_layers` must be a positive integer, got {num_layers!r}")

        # ── Resolve kernel sizes ─────────────────────────────────────
        if isinstance(kernel_size, int):
            # Same kernel for every layer
            kernel_sizes = [kernel_size] * num_layers
        elif isinstance(kernel_size, (list, tuple)):
            if len(kernel_size) != num_layers:
                raise ValueError(
                    f"`kernel_size` list length ({len(kernel_size)}) "
                    f"must match `num_layers` ({num_layers})"
                )
            kernel_sizes = list(kernel_size)
        else:
            raise TypeError(
                f"`kernel_size` must be an int or list of ints, got {type(kernel_size).__name__!r}"
            )

        # Validate each individual kernel size (must be positive odd integer)
        for i, k in enumerate(kernel_sizes):
            if not isinstance(k, int) or k < 1 or k % 2 == 0:
                raise ValueError(
                    f"Each kernel size must be a positive odd integer. "
                    f"Got kernel_sizes[{i}] = {k!r}"
                )

        # ── Build the layer stack ────────────────────────────────────
        # Channel progression:
        #   Layer 1      : 1                              → base_filters
        #   Layer 2..N-1 : base_filters * 2^(i-1)        → base_filters * 2^i
        #   Last layer   : <previous out_channels>        → 1  (single channel output)

        layers = []
        in_channels = 1
        out_channels = base_filters

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            k = kernel_sizes[i]
            padding = k // 2   # 'same' padding — preserves H and W dimensions

            if is_last:
                out_channels = 1
            
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=padding))
            layers.append(nn.ReLU())

            in_channels = out_channels
            if not is_last:
                out_channels = out_channels * 2   # double filters for next layer

        self.network = nn.Sequential(*layers)

        # Store config for easy inspection
        self.num_layers   = num_layers
        self.kernel_sizes = kernel_sizes
        self.base_filters = base_filters

    def forward(self, x):
        return self.network(x)

    def __repr__(self):
        lines = [
            f"CNNModel(",
            f"  num_layers  = {self.num_layers}",
            f"  kernel_sizes= {self.kernel_sizes}",
            f"  base_filters= {self.base_filters}",
        ]
        for name, module in self.network.named_children():
            lines.append(f"  ({name}): {module}")
        lines.append(")")
        return "\n".join(lines)
