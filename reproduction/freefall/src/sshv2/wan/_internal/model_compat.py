"""Checkpoint-compatible bidirectional Wan DiT implementation.

This variant preserves the historical repository initialization and raw
forward path while living under the single shared Wan package.
"""

import torch
import torch.nn as nn
from typing import Literal
from diffsynth.models.wan_video_dit import *


# MARK: DiT Block

class DiTBlock(nn.Module):
    def __init__(
        self,
        has_text_input: bool,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.has_text_input = has_text_input
        self.has_image_input = has_image_input

        self.self_attn = SelfAttention(dim, num_heads, eps)
        if has_text_input or has_image_input:
            self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, mechanism=None, layer_id=None, vae_local=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        key_delta = None if mechanism is None else mechanism.key_delta(layer_id, x)
        replacement = None if mechanism is None else mechanism.key_value_replacement(layer_id, x)
        if vae_local is not None:
            local_replacement = vae_local.key_value_replacement(layer_id, x)
            if local_replacement is not None:
                replacement = local_replacement
        # The stock Wan attention has the two-argument interface.  Raw DiT
        # runs must retain that exact path; passing dormant mechanism kwargs
        # is not harmless because the upstream module rightfully rejects them.
        if key_delta is None and replacement is None:
            attn = self.self_attn(input_x, freqs)
        else:
            attn = self.self_attn(input_x, freqs, key_delta=key_delta, key_value_replacement=replacement)
        x = self.gate(x, gate_msa, attn)
        if self.has_text_input or self.has_image_input:
            x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp = self.ffn(input_x)
        if mechanism is not None: mlp = mlp + mechanism.response_delta(layer_id, input_x)
        if vae_local is not None: mlp = mlp + vae_local.response_delta(layer_id, input_x)
        x = self.gate(x, gate_mlp, mlp)
        return x


def build_fixed_spatial_pos_emb(size: int, dim: int):
    positions = torch.arange(size, dtype=torch.float32)
    return sinusoidal_embedding_1d(dim, positions)


class MotionMechanismAdapter(nn.Module):
    """Zero-gated, object-tube conditioned K/late-MLP adapters."""
    def __init__(self, dim, kind, rank):
        super().__init__(); self.kind=kind; self.dim=dim
        self.summary=nn.Sequential(nn.Linear(dim*3, rank),nn.SiLU(),nn.Linear(rank,dim))
        self.k=nn.Linear(dim,dim,bias=False); self.a=nn.Linear(dim+dim,rank,bias=False); self.b=nn.Linear(rank,dim,bias=False)
        self.gk=nn.Parameter(torch.zeros(())); self.gr=nn.Parameter(torch.zeros(())); self.motion=None; self.future=None; self.tube=None; self.canonical=None
        self.canonical_encoder=nn.Sequential(nn.Linear(4, rank),nn.SiLU(),nn.Linear(rank,dim))
        self.canonical_k=nn.Linear(dim,dim,bias=False); self.canonical_v=nn.Linear(dim,dim,bias=False)
        nn.init.normal_(self.k.weight, std=0.02); nn.init.normal_(self.b.weight, std=0.02)
        nn.init.normal_(self.canonical_k.weight, std=0.02); nn.init.normal_(self.canonical_v.weight, std=0.02)
        if self.kind.startswith("forced_"):
            # LoRA-style response initialization: A is random, B starts at
            # zero.  Unlike the earlier optional branch there is no scalar
            # gate; the source replacement below makes c mandatory.
            nn.init.normal_(self.a.weight, std=0.02); nn.init.zeros_(self.b.weight)
    def prepare(self, x, tube, canonical_motion=None):
        if self.kind=="none" or tube is None: self.motion=None; self.canonical=None; return
        b,_,d=x.shape; tube=tube.to(x.device).bool(); a=[]
        for t in (0,1):
            z=x[:,t*64:(t+1)*64]; m=tube[:,t].reshape(b,64).unsqueeze(-1); a.append((z*m).sum(1)/m.sum(1).clamp_min(1))
        if self.kind=="shuffled_motion": a[1]=a[1].roll(1,0)
        self.motion=self.summary(torch.cat([a[0],a[1],a[1]-a[0]],-1)); self.future=torch.arange(128,x.shape[1],device=x.device); self.tube=tube
        if self.kind in ("forced_canonical", "forced_global", "forced_shuffled"):
            if canonical_motion is None: raise ValueError("forced canonical bottleneck requires canonical_motion")
            self.canonical=self.canonical_encoder((canonical_motion.to(x.device,dtype=x.dtype)*10).to(dtype=x.dtype))
    def key_delta(self, layer, x):
        if self.motion is None or self.kind not in ("k_routing","combined","shuffled_motion") or layer not in range(15,20): return None
        z=x.new_zeros(x.shape); z[:,:128]=self.gk*self.k(self.motion).unsqueeze(1); return z
    def key_value_replacement(self, layer, x):
        if not self.kind.startswith("forced_") or layer not in range(15,20): return None
        # Every prefix trajectory source is replaced by the same canonical
        # K/V token.  Raw source content cannot traverse this direct route.
        b,s,_=x.shape; mask=torch.zeros((b,s),dtype=torch.bool,device=x.device); mask[:,:128]=self.tube.reshape(b,128)
        if self.kind == "forced_zero":
            k=x.new_zeros((b,s,self.dim)); v=x.new_zeros((b,s,self.dim))
        else:
            k=self.canonical_k(self.canonical).unsqueeze(1).expand(-1,s,-1)
            v=self.canonical_v(self.canonical).unsqueeze(1).expand(-1,s,-1)
        return mask,k,v
    def response_delta(self, layer, h):
        if self.kind in ("forced_canonical", "forced_global", "forced_shuffled"):
            if self.canonical is None or layer not in range(25,30): return 0
            z=h.new_zeros(h.shape); f=h[:,128:];c=self.canonical.unsqueeze(1).expand_as(f)
            z[:,128:]=0.1*self.b(torch.nn.functional.silu(self.a(torch.cat([f,c],-1)))); return z
        if self.motion is None or self.kind not in ("late_response","combined","generic_response") or layer not in range(25,30): return 0
        m=self.motion if self.kind!="generic_response" else self.motion.detach()*0
        z=h.new_zeros(h.shape); f=h[:,128:]; z[:,128:]=self.gr*self.b(torch.nn.functional.silu(self.a(torch.cat([f,m.unsqueeze(1).expand_as(f)],-1)))); return z


class LocalCanonicalInteraction(nn.Module):
    """Shared wall-local interaction computation with soft global writeback.

    It is deliberately geometry-only: phi selects a coordinate chart, while
    the same small spatiotemporal module is reused at all four depths and for
    both walls.  No velocity, normal, alpha, or future simulation value enters
    this module.
    """
    def __init__(self, dim, heads, mode, roi, size, layers):
        super().__init__()
        self.mode, self.roi, self.size, self.layers = mode, tuple(roi), tuple(size), tuple(layers)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.time = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        # Identity at initialization for residual mode; gradients first reach
        # the output projection, then the shared interaction body.
        nn.init.zeros_(self.out.weight)
        self.geometry = None

    def prepare(self, geometry):
        self.geometry = geometry

    def _update(self, u, t):
        b, f, h, w, d = u.shape
        q = u.reshape(b, f * h * w, d)
        qn = self.norm(q)
        a = self.attn(qn, qn, qn, need_weights=False)[0]
        q = q + a + self.ffn(q + self.time(t).unsqueeze(1))
        return u + .1 * self.out(q).reshape(b, f, h, w, d)

    def apply(self, x, layer, t, grid_size):
        if self.geometry is None or layer not in self.layers:
            return x
        from einops import rearrange
        raise RuntimeError(
            "The removed local-canonical prototype is not supported"
        )
        import torch.nn.functional as F
        b, f, h, w = x.shape[0], *grid_size
        oh, ow = self.size
        # extract global hidden map into the common canonical ROI
        g = local_grid(self.geometry, oh, ow, self.roi, inverse=False)
        maps = rearrange(x, 'b (f h w) d -> (b f) d h w', f=f, h=h, w=w)
        g = g[:, None].expand(b, f, oh, ow, 2).reshape(b * f, oh, ow, 2)
        u = F.grid_sample(maps.float(), g, mode='bilinear', padding_mode='border', align_corners=False)
        u = rearrange(u, '(b f) d h w -> b f h w d', b=b, f=f).to(dtype=x.dtype)
        updated = self._update(u, t)
        # Sample the local update back onto every global token location.  The
        # feather is defined in canonical coordinates, so its boundary is
        # smooth even when the wall is rotated in the global scene.
        delta = updated - u
        local_maps = rearrange(delta, 'b f h w d -> (b f) d h w')
        back = inverse_local_grid(self.geometry, h, w, self.roi)
        back = back[:, None].expand(b, f, h, w, 2).reshape(b * f, h, w, 2)
        delta_global = F.grid_sample(local_maps.float(), back, mode='bilinear', padding_mode='zeros', align_corners=False)
        delta_global = rearrange(delta_global, '(b f) d h w -> b (f h w) d', b=b, f=f).to(dtype=x.dtype)
        mask = local_feather_mask(self.geometry, h, w, self.roi, feather=1 / max(oh, ow))
        mask = mask.expand(-1, f, -1, -1).reshape(b, f * h * w, 1).to(dtype=x.dtype)
        if self.mode == 'local_residual':
            return x + mask * delta_global
        # Forced use suppresses the native feature inside the ROI and writes
        # back the complete local state (rather than a rotated hard patch).
        full_local = rearrange(updated, 'b f h w d -> (b f) d h w')
        global_state = F.grid_sample(full_local.float(), back, mode='bilinear', padding_mode='zeros', align_corners=False)
        global_state = rearrange(global_state, '(b f) d h w -> b (f h w) d', b=b, f=f).to(dtype=x.dtype)
        return (1 - mask) * x + mask * global_state


class FrozenConv1PrefixConditioner(nn.Module):
    """Mandatory K/V path from a frozen Wan-conv1 local prefix crop.

    The crop is made offline from RGB frames 0..4 only, centred on the visible
    wall contact and rotated into its wall frame.  The small trainable map is
    merely an interface into the DiT; it has no time access beyond those five
    input frames and never receives a simulator velocity or future label.
    """
    def __init__(self, dim: int, channels: int = 96, frames: int = 5, size: int = 8, rank: int = 128):
        super().__init__()
        self.input_norm = nn.LayerNorm(channels * frames)
        self.summary = nn.Sequential(nn.Linear(channels * frames, rank), nn.SiLU(), nn.Linear(rank, dim))
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.a = nn.Linear(dim * 2, rank, bias=False)
        self.b = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.k.weight, std=.02)
        nn.init.normal_(self.v.weight, std=.02)
        nn.init.normal_(self.a.weight, std=.02)
        nn.init.zeros_(self.b.weight)  # LoRA-style non-destructive start.
        self.state = None

    def prepare(self, crop: torch.Tensor | None):
        if crop is None:
            self.state = None
            return
        # [B,96,5,8,8] -> temporal channel trajectory.  Spatial averaging is
        # deliberately after the object-centred canonical crop; it preserves
        # the five-frame trajectory while removing global chart coordinates.
        crop = crop.to(dtype=self.input_norm.weight.dtype).mean((-1, -2)).flatten(1)
        self.state = self.summary(self.input_norm(crop))

    def key_value_replacement(self, layer: int, x: torch.Tensor):
        if self.state is None or layer not in range(15, 20):
            return None
        b, tokens, dim = x.shape
        # All conditioned prefix tokens become an explicit shared local state.
        # Future tokens and all non-prefix paths are untouched.
        mask = torch.zeros((b, tokens), dtype=torch.bool, device=x.device)
        mask[:, :128] = True
        state = self.state.to(dtype=x.dtype, device=x.device)
        return mask, self.k(state).unsqueeze(1).expand(-1, tokens, -1), self.v(state).unsqueeze(1).expand(-1, tokens, -1)

    def response_delta(self, layer: int, h: torch.Tensor):
        if self.state is None or layer not in range(25, 30):
            return 0
        out = h.new_zeros(h.shape)
        state = self.state.to(dtype=h.dtype, device=h.device).unsqueeze(1).expand(-1, h.shape[1] - 128, -1)
        out[:, 128:] = .1 * self.b(torch.nn.functional.silu(self.a(torch.cat([h[:, 128:], state], -1))))
        return out


# MARK: Wan Model

class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_text_input: bool,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        spatial_pos_emb_size: tuple[int, int] = (128, 128),
        spatial_pos_emb_mode: Literal["rope", "learned", "absolute"] = "rope",
        mechanism_adapter: str = "none", mechanism_rank: int = 32,
        trajectory_renderer: bool = False, trajectory_channels: int = 1, renderer_static_context: bool = False, renderer_context_exact: bool = False,
        latent_canonical_mode: str = "none", local_canonical_roi=(.18,.84,.39,1.),
        local_canonical_size=(6,6), local_canonical_layers=(7,14,22,29),
        vae_local_conditioning: str = "none", vae_local_condition_variant: str = "tight_p1",
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_text_input = has_text_input
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.spatial_pos_emb_size = spatial_pos_emb_size
        self.spatial_pos_emb_mode = spatial_pos_emb_mode
        self.mechanism_adapter = MotionMechanismAdapter(dim, mechanism_adapter, mechanism_rank)
        self.trajectory_renderer = trajectory_renderer
        self.renderer_static_context = renderer_static_context
        self.renderer_context_exact = renderer_context_exact
        self.latent_canonical_mode = latent_canonical_mode
        self.vae_local_conditioning = vae_local_conditioning
        self.vae_local = FrozenConv1PrefixConditioner(dim) if vae_local_conditioning == "forced_kv" else None
        self.local_canonical = None if latent_canonical_mode not in ('local_residual', 'local_forced') else LocalCanonicalInteraction(
            dim, num_heads, latent_canonical_mode, local_canonical_roi, local_canonical_size, local_canonical_layers)
        self.trajectory_map_embed = nn.Conv3d(trajectory_channels, dim, kernel_size=(1,2,2), stride=(1,2,2), bias=False) if trajectory_renderer else None
        if self.trajectory_map_embed is not None: nn.init.normal_(self.trajectory_map_embed.weight, std=0.02)

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        if has_text_input:
            self.text_embedding = nn.Sequential(
                nn.Linear(text_dim, dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim)
            )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_text_input, has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        if spatial_pos_emb_mode not in {"rope", "learned", "absolute"}:
            raise ValueError(f"Expected `spatial_pos_emb_mode` to be one of ('rope', 'learned', 'absolute'), but got {spatial_pos_emb_mode!r}.")
        if spatial_pos_emb_mode != "rope":
            if spatial_pos_emb_size[0] < 1 or spatial_pos_emb_size[1] < 1:
                raise ValueError(f"Expected positive spatial_pos_emb_size, got {spatial_pos_emb_size}.")
            self.freqs_temporal = precompute_freqs_cis(head_dim)
            if spatial_pos_emb_mode == "learned":
                self.spatial_pos_emb_h = nn.Parameter(torch.zeros(spatial_pos_emb_size[0], dim))
                self.spatial_pos_emb_w = nn.Parameter(torch.zeros(spatial_pos_emb_size[1], dim))
                nn.init.zeros_(self.spatial_pos_emb_h)
                nn.init.zeros_(self.spatial_pos_emb_w)
            else:
                self.register_buffer(
                    "spatial_pos_emb_h",
                    build_fixed_spatial_pos_emb(spatial_pos_emb_size[0], dim),
                    persistent=False
                )
                self.register_buffer(
                    "spatial_pos_emb_w",
                    build_fixed_spatial_pos_emb(spatial_pos_emb_size[1], dim),
                    persistent=False
                )

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: torch.Tensor | None = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
