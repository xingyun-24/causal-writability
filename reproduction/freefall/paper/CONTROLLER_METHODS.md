# Free Fall PCA and Controller Methods

The PCA basis is fitted from 64 training pairs only. The 64 held-out pairs never enter the basis, rank selection, scale, controller fit, or intervention-site selection. Rank 2 is the smallest rank retaining at least 99% of training residual energy.

The official donor-free controller is direction-specific:

```text
z_hat^(r)(g_target) = beta_0^(r) + beta_g^(r) g_target,  r in {low, high}
```

It predicts two coordinates using target direction and target gravity only. Both oracle and controller use the same train-only PCA basis, global fit-only scale `1.002281550`, after-Block-1 site, condition-prefix token scope, 20 flow-matching calls, generation seed, and evaluator.

Held-out coordinate joint R2 is `0.997010` for low and `0.992003` for high. The top-2 oracle has E3 `96.8750%` and the fit-only controller has E3 `95.3125%` on 64 held-out receivers.
