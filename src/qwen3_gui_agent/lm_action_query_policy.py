from __future__ import annotations

import os

import torch
from torch import nn



class _JointQueryBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        *,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_count = int(queries.shape[1])
        causal_mask = torch.triu(
            torch.ones(
                query_count,
                query_count,
                device=queries.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        attended, _ = self.self_attention(
            queries,
            queries,
            queries,
            attn_mask=causal_mask,
            need_weights=False,
        )
        queries = self.self_norm(queries + attended)
        attended, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=~memory_mask.bool(),
            need_weights=False,
        )
        queries = self.cross_norm(queries + attended)
        return self.ffn_norm(queries + self.ffn(queries))


class _DualStreamQueryBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.context_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.visual_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.stream_gate = nn.Linear(hidden_dim, 2)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        *,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        context_token_count: int,
    ) -> torch.Tensor:
        query_count = int(queries.shape[1])
        causal_mask = torch.triu(
            torch.ones(
                query_count,
                query_count,
                device=queries.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        attended, _ = self.self_attention(
            queries,
            queries,
            queries,
            attn_mask=causal_mask,
            need_weights=False,
        )
        queries = self.self_norm(queries + attended)
        context_count = max(0, min(int(context_token_count), int(memory.shape[1])))
        if context_count <= 0:
            context = memory
            context_mask = memory_mask
        else:
            context = memory[:, :context_count]
            context_mask = memory_mask[:, :context_count]
        visual = memory[:, context_count:]
        visual_mask = memory_mask[:, context_count:]
        if int(visual.shape[1]) <= 0:
            visual = context
            visual_mask = context_mask
        context_attended, _ = self.context_attention(
            queries,
            context,
            context,
            key_padding_mask=~context_mask.bool(),
            need_weights=False,
        )
        visual_attended, _ = self.visual_attention(
            queries,
            visual,
            visual,
            key_padding_mask=~visual_mask.bool(),
            need_weights=False,
        )
        gates = torch.softmax(self.stream_gate(queries).float(), dim=-1).to(queries.dtype)
        attended = gates[..., :1] * context_attended + gates[..., 1:] * visual_attended
        queries = self.cross_norm(queries + attended)
        return self.ffn_norm(queries + self.ffn(queries))


class _TwoWayContextVisualBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.context_to_visual = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.visual_to_context = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.context_ffn_norm = nn.LayerNorm(hidden_dim)
        self.visual_ffn_norm = nn.LayerNorm(hidden_dim)
        self.context_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.visual_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        context: torch.Tensor,
        visual: torch.Tensor,
        *,
        context_mask: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended_context, _ = self.context_to_visual(
            context,
            visual,
            visual,
            key_padding_mask=~visual_mask.bool(),
            need_weights=False,
        )
        context = self.context_norm(context + attended_context)
        context = self.context_ffn_norm(context + self.context_ffn(context))
        attended_visual, _ = self.visual_to_context(
            visual,
            context,
            context,
            key_padding_mask=~context_mask.bool(),
            need_weights=False,
        )
        visual = self.visual_norm(visual + attended_visual)
        visual = self.visual_ffn_norm(visual + self.visual_ffn(visual))
        return context, visual


class _ZeroInitRoleProjection(nn.Module):
    """Role-specific residual projection that preserves old checkpoints at init."""

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        bottleneck_dim = max(1, int(bottleneck_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class _ActionResidualRouter(nn.Module):
    """Decode action memory separately for type routing and A-to-visual grounding."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        action_type_count: int,
        coord_bins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.type_action_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.coordinate_action_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.coordinate_visual_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.type_action_norm = nn.LayerNorm(hidden_dim)
        self.coordinate_action_norm = nn.LayerNorm(hidden_dim)
        self.coordinate_visual_norm = nn.LayerNorm(hidden_dim)
        self.type_ffn_norm = nn.LayerNorm(hidden_dim)
        self.coordinate_ffn_norm = nn.LayerNorm(hidden_dim)
        self.type_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.coordinate_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.type_delta_head = nn.Linear(hidden_dim, action_type_count)
        self.pointer_router_head = nn.Linear(hidden_dim, 2)
        self.x_delta_head = nn.Linear(hidden_dim, coord_bins)
        self.y_delta_head = nn.Linear(hidden_dim, coord_bins)
        for head in (
            self.type_delta_head,
            self.pointer_router_head,
            self.x_delta_head,
            self.y_delta_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _safe_role_mask(role_mask: torch.Tensor, memory_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        role_mask = role_mask.bool() & memory_mask.bool()
        valid = role_mask.any(dim=1)
        safe_mask = role_mask.clone()
        for row_index in torch.nonzero(~valid, as_tuple=False).flatten().tolist():
            fallback = torch.nonzero(memory_mask[row_index].bool(), as_tuple=False).flatten()
            if int(fallback.numel()) > 0:
                safe_mask[row_index, fallback[0]] = True
        return safe_mask, valid

    def _attend_action(
        self,
        query: torch.Tensor,
        *,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        action_mask: torch.Tensor,
        coordinate: bool,
    ) -> torch.Tensor:
        safe_mask, valid = self._safe_role_mask(action_mask, memory_mask)
        attention = (
            self.coordinate_action_attention if coordinate else self.type_action_attention
        )
        attended, _ = attention(
            query.unsqueeze(1),
            memory,
            memory,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        attended = attended[:, 0] * valid.unsqueeze(-1).to(attended.dtype)
        norm = self.coordinate_action_norm if coordinate else self.type_action_norm
        return norm(query + attended)

    def type_state(
        self,
        query: torch.Tensor,
        *,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        state = self._attend_action(
            query,
            memory=memory,
            memory_mask=memory_mask,
            action_mask=action_mask,
            coordinate=False,
        )
        return self.type_ffn_norm(state + self.type_ffn(state))

    def coordinate_state(
        self,
        query: torch.Tensor,
        *,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        action_mask: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        state = self._attend_action(
            query,
            memory=memory,
            memory_mask=memory_mask,
            action_mask=action_mask,
            coordinate=True,
        )
        safe_visual, valid_visual = self._safe_role_mask(visual_mask, memory_mask)
        attended, _ = self.coordinate_visual_attention(
            state.unsqueeze(1),
            memory,
            memory,
            key_padding_mask=~safe_visual,
            need_weights=False,
        )
        attended = attended[:, 0] * valid_visual.unsqueeze(-1).to(attended.dtype)
        state = self.coordinate_visual_norm(state + attended)
        return self.coordinate_ffn_norm(state + self.coordinate_ffn(state))


class LMActionQueryPolicy(nn.Module):
    """Lightweight policy over causal LM states before Type, X, and Y."""

    MEMORY_ROLES = (
        "reasoning",
        "action",
        "post",
        "actionctx",
        "boundary",
        "sidecar",
        "visual",
    )
    QUERY_ROLES = ("type", "x", "y")

    def __init__(
        self,
        *,
        input_dim: int,
        action_type_count: int,
        terminate_count: int,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        coord_bins: int = 512,
        layer_count: int = 1,
        dropout: float = 0.0,
        teacher_forcing_ratio: float = 0.5,
        interaction_mode: str | None = None,
        interaction_depth: int | None = None,
        layer_mix_mode: str | None = None,
        pointer_action_indices: list[int] | tuple[int, ...] | None = None,
        context_role_mode: str | None = None,
        latent_role_layout: str | None = None,
        router_mode: str | None = None,
        coordinate_prior_mode: str | None = None,
        coordinate_prior_gate_init: float | None = None,
        role_projection_mode: str | None = None,
        residual_router_mode: str | None = None,
        role_adapter_dim: int | None = None,
        action_adapter_dim: int | None = None,
        residual_gate_init: float | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.coord_bins = max(16, int(coord_bins))
        self.layer_count = max(1, int(layer_count))
        self.teacher_forcing_ratio = min(1.0, max(0.0, float(teacher_forcing_ratio)))
        self.interaction_mode = str(
            interaction_mode
            or os.environ.get("LARA_ACTION_QUERY_INTERACTION_MODE", "independent")
        ).strip().lower()
        if self.interaction_mode not in {"independent", "joint", "dual_stream", "two_way"}:
            raise ValueError(f"Unsupported Action-Query interaction mode: {self.interaction_mode}")
        self.interaction_depth = max(
            1,
            int(
                interaction_depth
                if interaction_depth is not None
                else os.environ.get("LARA_ACTION_QUERY_INTERACTION_DEPTH", "1")
            ),
        )
        self.layer_mix_mode = str(
            layer_mix_mode
            or os.environ.get("LARA_ACTION_QUERY_LAYER_MIX_MODE", "shared")
        ).strip().lower()
        if self.layer_mix_mode not in {
            "shared",
            "semantic_visual",
            "query_specific",
            "fully_separate",
        }:
            raise ValueError(
                f"Unsupported Action-Query layer mix mode: {self.layer_mix_mode}"
            )
        self.context_role_mode = str(
            context_role_mode
            or os.environ.get("LARA_ACTION_QUERY_CONTEXT_ROLE_MODE", "flat")
        ).strip().lower()
        if self.context_role_mode not in {"flat", "role_gated", "role_branch"}:
            raise ValueError(
                f"Unsupported Action-Query context role mode: {self.context_role_mode}"
            )
        self.latent_role_layout = str(
            latent_role_layout
            or os.environ.get("LARA_ACTION_QUERY_LATENT_ROLE_LAYOUT", "all_reasoning")
        ).strip().lower()
        if self.latent_role_layout not in {"all_reasoning", "last_action"}:
            raise ValueError(
                f"Unsupported Action-Query latent role layout: {self.latent_role_layout}"
            )
        self.router_mode = str(
            router_mode
            or os.environ.get("LARA_ACTION_QUERY_ROUTER_MODE", "none")
        ).strip().lower()
        if self.router_mode not in {"none", "pointer_gate"}:
            raise ValueError(f"Unsupported Action-Query router mode: {self.router_mode}")
        self.role_projection_mode = str(
            role_projection_mode
            or os.environ.get("LARA_ACTION_QUERY_ROLE_PROJECTION_MODE", "shared")
        ).strip().lower()
        if self.role_projection_mode not in {"shared", "independent_residual"}:
            raise ValueError(
                f"Unsupported Action-Query role projection mode: {self.role_projection_mode}"
            )
        self.residual_router_mode = str(
            residual_router_mode
            or os.environ.get("LARA_ACTION_RESIDUAL_ROUTER_MODE", "off")
        ).strip().lower()
        if self.residual_router_mode not in {"off", "action_to_visual"}:
            raise ValueError(
                f"Unsupported Action-Query residual router mode: {self.residual_router_mode}"
            )
        self.coordinate_prior_mode = str(
            coordinate_prior_mode
            or os.environ.get("LARA_ACTION_QUERY_COORDINATE_PRIOR_MODE", "none")
        ).strip().lower()
        if self.coordinate_prior_mode not in {
            "none",
            "latent_action_logits",
            "predicted_coordinate_logits",
            "lm_action_slot_logits",
        }:
            raise ValueError(
                "Unsupported Action-Query coordinate prior mode: "
                f"{self.coordinate_prior_mode}"
            )
        configured_sources = str(
            os.environ.get(
                "LARA_ACTION_QUERY_CONTEXT_SOURCES",
                ",".join(self.MEMORY_ROLES),
            )
        )
        self.context_sources = {
            item.strip().lower() for item in configured_sources.split(",") if item.strip()
        }
        unknown_sources = self.context_sources.difference(self.MEMORY_ROLES)
        if unknown_sources:
            raise ValueError(f"Unsupported Action-Query context sources: {sorted(unknown_sources)}")
        if not self.context_sources:
            raise ValueError("At least one Action-Query context source must be enabled.")
        self.layer_logits = nn.Parameter(torch.zeros(self.layer_count, dtype=torch.float32))
        self.semantic_layer_logits = nn.Parameter(
            torch.zeros(self.layer_count, dtype=torch.float32)
        )
        self.visual_layer_logits = nn.Parameter(
            torch.zeros(self.layer_count, dtype=torch.float32)
        )
        self.type_layer_logits = nn.Parameter(
            torch.zeros(self.layer_count, dtype=torch.float32)
        )
        self.x_layer_logits = nn.Parameter(
            torch.zeros(self.layer_count, dtype=torch.float32)
        )
        self.y_layer_logits = nn.Parameter(
            torch.zeros(self.layer_count, dtype=torch.float32)
        )
        self.query_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        self.context_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        self.visual_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        role_bottleneck = int(
            role_adapter_dim
            if role_adapter_dim is not None
            else os.environ.get("LARA_ACTION_QUERY_ROLE_ADAPTER_DIM", "512")
        )
        action_bottleneck = int(
            action_adapter_dim
            if action_adapter_dim is not None
            else os.environ.get("LARA_ACTION_QUERY_ACTION_ADAPTER_DIM", "2048")
        )
        self.role_context_adapters = nn.ModuleDict(
            {
                "reasoning": _ZeroInitRoleProjection(
                    input_dim, hidden_dim, role_bottleneck
                ),
                "boundary": _ZeroInitRoleProjection(
                    input_dim, hidden_dim, role_bottleneck
                ),
                "action": _ZeroInitRoleProjection(
                    input_dim, hidden_dim, action_bottleneck
                ),
            }
        )
        configured_train_roles = str(
            os.environ.get(
                "LARA_ACTION_RESIDUAL_TRAIN_ROLES",
                "reasoning,boundary,action",
            )
        )
        self.trainable_residual_roles = {
            item.strip().lower()
            for item in configured_train_roles.split(",")
            if item.strip()
        }
        supported_train_roles = set(self.role_context_adapters.keys())
        unknown_train_roles = self.trainable_residual_roles.difference(
            supported_train_roles
        )
        if unknown_train_roles:
            raise ValueError(
                "Unsupported trainable residual roles: "
                f"{sorted(unknown_train_roles)}"
            )
        if not self.trainable_residual_roles:
            raise ValueError("At least one residual role adapter must be trainable.")
        self.memory_type = nn.Parameter(torch.zeros(2, hidden_dim))
        self.memory_role_embedding = nn.Parameter(
            torch.zeros(len(self.MEMORY_ROLES), hidden_dim)
        )
        self.query_role_logits = nn.Parameter(
            torch.zeros(len(self.QUERY_ROLES), len(self.MEMORY_ROLES), dtype=torch.float32)
        )
        self.visual_position = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        block_type = _DualStreamQueryBlock if self.interaction_mode == "dual_stream" else _JointQueryBlock
        self.interaction_blocks = nn.ModuleList(
            block_type(hidden_dim, num_heads, dropout)
            for _ in range(self.interaction_depth)
        ) if self.interaction_mode != "independent" else nn.ModuleList()
        self.context_visual_blocks = nn.ModuleList(
            _TwoWayContextVisualBlock(hidden_dim, num_heads, dropout)
            for _ in range(self.interaction_depth)
        ) if self.interaction_mode == "two_way" else nn.ModuleList()
        self.action_type_head = nn.Linear(hidden_dim, action_type_count)
        self.pointer_router_head = nn.Linear(hidden_dim, 2)
        self.x_head = nn.Linear(hidden_dim, self.coord_bins)
        self.y_head = nn.Linear(hidden_dim, self.coord_bins)
        self.terminate_head = nn.Linear(hidden_dim, terminate_count)
        self.action_residual_router = _ActionResidualRouter(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            action_type_count=action_type_count,
            coord_bins=self.coord_bins,
            dropout=dropout,
        )
        residual_init = float(
            residual_gate_init
            if residual_gate_init is not None
            else os.environ.get("LARA_ACTION_RESIDUAL_GATE_INIT", "-4.0")
        )
        self.action_residual_gate_logits = nn.Parameter(
            torch.full((4,), residual_init, dtype=torch.float32)
        )
        # Start close to the proven Head-only behavior. Training can increase
        # either gate only when the frozen Stage2 coordinate prior is useful.
        gate_init = float(
            coordinate_prior_gate_init
            if coordinate_prior_gate_init is not None
            else os.environ.get("LARA_ACTION_QUERY_COORDINATE_PRIOR_GATE_INIT", "-2.0")
        )
        self.coordinate_prior_gate_logits = nn.Parameter(
            torch.full((2,), gate_init, dtype=torch.float32)
        )
        self.action_type_prior_gate_logit = nn.Parameter(
            torch.tensor(gate_init, dtype=torch.float32)
        )
        nn.init.normal_(self.memory_type, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_role_embedding, mean=0.0, std=0.02)
        pointer_indices = {
            int(index)
            for index in (pointer_action_indices or ())
            if 0 <= int(index) < int(action_type_count)
        }
        action_pointer_groups = torch.zeros(action_type_count, dtype=torch.long)
        for index in pointer_indices:
            action_pointer_groups[index] = 1
        self.register_buffer(
            "action_pointer_groups",
            action_pointer_groups,
            persistent=True,
        )

    @property
    def action_residual_router_enabled(self) -> bool:
        return self.residual_router_mode == "action_to_visual"


    def _apply_role_projection_residuals(
        self,
        context: torch.Tensor,
        context_states: torch.Tensor,
        context_role_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.role_projection_mode != "independent_residual":
            return context
        for role_name, adapter in self.role_context_adapters.items():
            role_index = self.MEMORY_ROLES.index(role_name)
            role_mask = context_role_ids.eq(role_index).unsqueeze(-1).to(context.dtype)
            context = context + role_mask * adapter(context_states)
        return context

    def _action_residual_masks(
        self,
        *,
        memory_mask: torch.Tensor,
        memory_role_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_mask = memory_role_ids.eq(self.MEMORY_ROLES.index("action"))
        visual_mask = memory_role_ids.eq(self.MEMORY_ROLES.index("visual"))
        if not bool((action_mask & memory_mask.bool()).any(dim=1).all().item()):
            raise ValueError("v20 action residual router requires Action memory for every sample.")
        return action_mask, visual_mask

    def fuse_action_type_residual(
        self,
        *,
        query_state: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_role_ids: torch.Tensor,
        action_type_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        gates = torch.sigmoid(self.action_residual_gate_logits.float()).to(
            device=action_type_logits.device,
            dtype=action_type_logits.dtype,
        )
        if not self.action_residual_router_enabled:
            return action_type_logits, None, gates.new_zeros(gates.shape)
        action_mask, _ = self._action_residual_masks(
            memory_mask=memory_mask,
            memory_role_ids=memory_role_ids,
        )
        type_state = self.action_residual_router.type_state(
            self.query_projection(query_state),
            memory=memory,
            memory_mask=memory_mask,
            action_mask=action_mask,
        )
        type_delta = self.action_residual_router.type_delta_head(type_state)
        pointer_logits = self.action_residual_router.pointer_router_head(type_state)
        group_log_probs = torch.log_softmax(pointer_logits.float(), dim=-1).to(
            action_type_logits.dtype
        )
        group_delta = group_log_probs.index_select(1, self.action_pointer_groups)
        group_delta = group_delta - group_delta.mean(dim=-1, keepdim=True)
        action_type_logits = (
            action_type_logits
            + gates[0] * type_delta
            + gates[1] * group_delta
        )
        return action_type_logits, pointer_logits, gates

    def fuse_action_coordinate_residual(
        self,
        *,
        x_query_state: torch.Tensor,
        y_query_state: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_role_ids: torch.Tensor,
        x_logits: torch.Tensor,
        y_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gates = torch.sigmoid(self.action_residual_gate_logits.float()).to(
            device=x_logits.device,
            dtype=x_logits.dtype,
        )
        if not self.action_residual_router_enabled:
            return x_logits, y_logits, gates.new_zeros(gates.shape)
        action_mask, visual_mask = self._action_residual_masks(
            memory_mask=memory_mask,
            memory_role_ids=memory_role_ids,
        )
        x_state = self.action_residual_router.coordinate_state(
            self.query_projection(x_query_state),
            memory=memory,
            memory_mask=memory_mask,
            action_mask=action_mask,
            visual_mask=visual_mask,
        )
        y_state = self.action_residual_router.coordinate_state(
            self.query_projection(y_query_state),
            memory=memory,
            memory_mask=memory_mask,
            action_mask=action_mask,
            visual_mask=visual_mask,
        )
        x_logits = x_logits + gates[2] * self.action_residual_router.x_delta_head(x_state)
        y_logits = y_logits + gates[3] * self.action_residual_router.y_delta_head(y_state)
        return x_logits, y_logits, gates






    def _source_enabled_mask(self, role_ids: torch.Tensor) -> torch.Tensor:
        enabled = torch.tensor(
            [role in self.context_sources for role in self.MEMORY_ROLES],
            device=role_ids.device,
            dtype=torch.bool,
        )
        return enabled.index_select(0, role_ids.reshape(-1)).reshape_as(role_ids)


    def _role_condition(
        self,
        *,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_role_ids: torch.Tensor | None,
        query_role: str,
    ) -> torch.Tensor:
        if self.context_role_mode != "role_gated" or memory_role_ids is None:
            return memory.new_zeros((int(memory.shape[0]), int(memory.shape[-1])))
        pooled_rows = []
        valid_roles = []
        for role_index in range(len(self.MEMORY_ROLES)):
            role_mask = memory_mask.bool() & memory_role_ids.eq(role_index)
            weights = role_mask.unsqueeze(-1).to(memory.dtype)
            pooled_rows.append(
                (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            )
            valid_roles.append(role_mask.any(dim=1))
        pooled = torch.stack(pooled_rows, dim=1)
        valid = torch.stack(valid_roles, dim=1)
        logits = self.query_role_logits[self.QUERY_ROLES.index(query_role)].unsqueeze(0)
        logits = logits.expand(int(memory.shape[0]), -1).masked_fill(~valid, -1e4)
        weights = torch.softmax(logits.float(), dim=-1).to(memory.dtype)
        return (pooled * weights.unsqueeze(-1)).sum(dim=1)

    def classify_action_type(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.action_type_head(state)
        if self.router_mode != "pointer_gate":
            return logits, None
        router_logits = self.pointer_router_head(state)
        group_log_probs = torch.log_softmax(router_logits.float(), dim=-1).to(logits.dtype)
        group_bias = group_log_probs.index_select(1, self.action_pointer_groups)
        return logits + group_bias, router_logits

    @staticmethod
    def _fallback_grid_size(token_count: int) -> tuple[int, int]:
        width = max(1, int(round(max(1, token_count) ** 0.5)))
        while width > 1 and token_count % width:
            width -= 1
        height = max(1, token_count // width)
        return (height, width) if height * width == token_count else (token_count, 1)

    def _layer_logits_for_role(self, role: str) -> torch.Tensor:
        role = str(role or "shared").strip().lower()
        if role not in {"shared", "semantic", "visual", "type", "x", "y"}:
            raise ValueError(f"Unsupported layer-mix role: {role}")
        if self.layer_mix_mode == "shared" or role == "shared":
            return self.layer_logits
        if self.layer_mix_mode == "semantic_visual":
            return self.visual_layer_logits if role == "visual" else self.semantic_layer_logits
        if self.layer_mix_mode == "query_specific":
            return {
                "type": self.type_layer_logits,
                "x": self.x_layer_logits,
                "y": self.y_layer_logits,
            }.get(role, self.layer_logits)
        return {
            "semantic": self.semantic_layer_logits,
            "visual": self.visual_layer_logits,
            "type": self.type_layer_logits,
            "x": self.x_layer_logits,
            "y": self.y_layer_logits,
        }.get(role, self.layer_logits)


    def mix_hidden_layers(
        self,
        hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        role: str = "shared",
    ) -> torch.Tensor:
        available = list(hidden_states)
        if not available:
            raise ValueError("At least one hidden-state layer is required.")
        selected = available[-min(self.layer_count, len(available)) :]
        if len(selected) == 1:
            return selected[0]
        logits = self._layer_logits_for_role(role)[-len(selected) :].to(
            device=selected[0].device,
            dtype=torch.float32,
        )
        weights = torch.softmax(logits, dim=0).to(dtype=selected[0].dtype)
        return sum(weight * state for weight, state in zip(weights, selected))

    def _visual_positions(
        self,
        *,
        batch_size: int,
        token_count: int,
        grid_sizes: list[tuple[int, int]] | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        rows = []
        for batch_index in range(batch_size):
            height, width = (
                grid_sizes[batch_index]
                if grid_sizes and batch_index < len(grid_sizes)
                else self._fallback_grid_size(token_count)
            )
            height, width = max(1, int(height)), max(1, int(width))
            if height * width > token_count:
                height, width = self._fallback_grid_size(token_count)
            positions = reference.new_full((token_count, 2), 0.5)
            valid_count = min(token_count, height * width)
            indices = torch.arange(valid_count, device=reference.device)
            positions[:valid_count, 0] = (indices.remainder(width).to(reference.dtype) + 0.5) / width
            positions[:valid_count, 1] = (
                torch.div(indices, width, rounding_mode="floor").to(reference.dtype) + 0.5
            ) / height
            rows.append(positions)
        return torch.stack(rows)

    def encode_memory(
        self,
        *,
        current_visual_tokens: torch.Tensor,
        current_visual_token_mask: torch.Tensor,
        target_patch_grid_sizes: list[tuple[int, int]] | None,
        context_states: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        context_role_ids: torch.Tensor | None = None,
        return_role_ids: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(current_visual_tokens.shape[0])
        visual = self.visual_projection(current_visual_tokens)
        positions = self._visual_positions(
            batch_size=batch_size,
            token_count=int(visual.shape[1]),
            grid_sizes=target_patch_grid_sizes,
            reference=visual,
        )
        visual = visual + self.memory_type[1] + self.visual_position(positions)
        visual_mask = current_visual_token_mask.bool()
        memory_rows = [visual]
        mask_rows = [visual_mask]
        role_rows = [
            torch.full(
                visual_mask.shape,
                self.MEMORY_ROLES.index("visual"),
                device=visual_mask.device,
                dtype=torch.long,
            )
        ]
        if context_states is not None and int(context_states.shape[1]) > 0:
            context = self.context_projection(context_states) + self.memory_type[0]
            if context_mask is None:
                context_mask = torch.ones(
                    context.shape[:2],
                    device=context.device,
                    dtype=torch.bool,
                )
            context_mask = context_mask.bool()
            if context_role_ids is None:
                context_role_ids = torch.full(
                    context_mask.shape,
                    self.MEMORY_ROLES.index("reasoning"),
                    device=context_mask.device,
                    dtype=torch.long,
                )
            else:
                context_role_ids = context_role_ids.to(
                    device=context_mask.device,
                    dtype=torch.long,
                )
            if tuple(context_role_ids.shape) != tuple(context_mask.shape):
                raise ValueError(
                    f"Context role ids must match context mask: "
                    f"{tuple(context_role_ids.shape)} vs {tuple(context_mask.shape)}"
                )
            context = self._apply_role_projection_residuals(
                context,
                context_states,
                context_role_ids,
            )
            if self.context_role_mode in {"role_gated", "role_branch"}:
                context = context + self.memory_role_embedding.index_select(
                    0, context_role_ids.reshape(-1)
                ).reshape_as(context)
            if self.interaction_mode == "two_way":
                for block in self.context_visual_blocks:
                    context, visual = block(
                        context,
                        visual,
                        context_mask=context_mask,
                        visual_mask=visual_mask,
                    )
                memory_rows = [context, visual]
                mask_rows = [context_mask, visual_mask]
                role_rows = [context_role_ids, role_rows[0]]
            else:
                memory_rows.insert(0, context)
                mask_rows.insert(0, context_mask)
                role_rows.insert(0, context_role_ids)
        memory = torch.cat(memory_rows, dim=1)
        memory_mask = torch.cat(mask_rows, dim=1)
        memory_role_ids = torch.cat(role_rows, dim=1)
        if self.context_role_mode in {"role_gated", "role_branch"}:
            visual_role_mask = memory_role_ids.eq(self.MEMORY_ROLES.index("visual"))
            memory = memory + visual_role_mask.unsqueeze(-1).to(memory.dtype) * self.memory_role_embedding[
                self.MEMORY_ROLES.index("visual")
            ]
        memory_mask = memory_mask & self._source_enabled_mask(memory_role_ids)
        if not bool(memory_mask.any(dim=1).all().item()):
            raise ValueError("Action-Query source masking removed every memory token for a sample.")
        if return_role_ids:
            return memory, memory_mask, memory_role_ids
        return memory, memory_mask


    def decode_state(
        self,
        query_state: torch.Tensor,
        *,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_role_ids: torch.Tensor | None = None,
        query_role: str = "type",
    ) -> torch.Tensor:
        query = self.query_projection(query_state)
        query = query + self._role_condition(
            memory=memory,
            memory_mask=memory_mask,
            memory_role_ids=memory_role_ids,
            query_role=query_role,
        )
        query = query.unsqueeze(1)
        if self.context_role_mode == "role_branch":
            if memory_role_ids is None:
                raise ValueError("role_branch requires memory role ids.")
            branch_rows = []
            valid_rows = []
            for role_index in range(len(self.MEMORY_ROLES)):
                role_mask = memory_mask.bool() & memory_role_ids.eq(role_index)
                valid = role_mask.any(dim=1)
                safe_mask = role_mask.clone()
                missing = ~valid
                if bool(missing.any().item()):
                    safe_mask[missing, 0] = True
                branch, _ = self.cross_attention(
                    query,
                    memory,
                    memory,
                    key_padding_mask=~safe_mask,
                    need_weights=False,
                )
                branch_rows.append(branch[:, 0] * valid.unsqueeze(-1).to(branch.dtype))
                valid_rows.append(valid)
            branches = torch.stack(branch_rows, dim=1)
            valid_roles = torch.stack(valid_rows, dim=1)
            role_logits = self.query_role_logits[
                self.QUERY_ROLES.index(query_role)
            ].unsqueeze(0)
            role_logits = role_logits.expand(int(memory.shape[0]), -1)
            role_logits = role_logits.masked_fill(~valid_roles, -1e4)
            role_weights = torch.softmax(role_logits.float(), dim=-1).to(branches.dtype)
            attended = (branches * role_weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        else:
            attended, _ = self.cross_attention(
                query,
                memory,
                memory,
                key_padding_mask=~memory_mask.bool(),
                need_weights=False,
            )
        state = self.attention_norm(query + attended)
        state = self.ffn_norm(state + self.ffn(state))
        return state[:, 0]

    def training_condition(self, teacher: torch.Tensor | None, prediction: torch.Tensor) -> torch.Tensor:
        if not self.training or teacher is None or self.teacher_forcing_ratio <= 0.0:
            return prediction
        if self.teacher_forcing_ratio >= 1.0:
            return teacher
        use_teacher = torch.rand(prediction.shape, device=prediction.device) < self.teacher_forcing_ratio
        return torch.where(use_teacher, teacher, prediction)
