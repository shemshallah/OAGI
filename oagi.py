#!/usr/bin/env python3
# oagi_engine_chat.py
# OAGI Engine — Self-Expressive, Recursive, with Autonomous AGI-Seeking + Chat Driver

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import sys

torch.manual_seed(42)


# -----------------------------
# Cognitive Kernel
# -----------------------------
class CognitiveKernel(nn.Module):
    def __init__(self, state_dim: int, emo_dim: int, mem_dim: int):
        super().__init__()
        self.state_dim = state_dim
        total_input = state_dim * 2 + emo_dim + mem_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_input, state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, state_dim),
            nn.LayerNorm(state_dim)
        )

    def forward(self, self_state: torch.Tensor, context: torch.Tensor,
                emotion: torch.Tensor, memory: torch.Tensor,
                plasticity: torch.Tensor = None) -> torch.Tensor:
        x = torch.cat([self_state, context, emotion, memory], dim=-1)
        update = self.mlp(x)
        if plasticity is not None:
            update = update * plasticity.unsqueeze(-1)
        return self_state + update


# -----------------------------
# Recursive Memory
# -----------------------------
class RecursiveMemory(nn.Module):
    def __init__(self, capacity: int, key_dim: int, value_dim: int):
        super().__init__()
        self.capacity = capacity
        self.register_buffer('keys', torch.zeros(capacity, key_dim))
        self.register_buffer('values', torch.zeros(capacity, value_dim))
        self.register_buffer('usage', torch.zeros(capacity))
        self.register_buffer('timestamps', torch.zeros(capacity))
        self.write_index = 0

    def write(self, key: torch.Tensor, value: torch.Tensor, importance: float = 1.0):
        idx = self.write_index % self.capacity
        self.keys[idx] = key.detach()
        self.values[idx] = value.detach()
        self.usage[idx] = importance
        self.timestamps[idx] = self.write_index
        self.write_index += 1

    def read(self, query: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        if self.write_index == 0:
            return torch.zeros(self.values.shape[1], device=query.device)
        used = min(self.write_index, self.capacity)
        k = self.keys[:used]
        v = self.values[:used]
        sim = F.cosine_similarity(query.unsqueeze(0), k, dim=-1)
        weights = F.softmax(sim, dim=0)
        return torch.sum(weights.unsqueeze(-1) * v, dim=0)

    def get_recent_context(self, n: int = 5) -> List[torch.Tensor]:
        if self.write_index == 0:
            return []
        used = min(self.write_index, self.capacity)
        recent_indices = torch.argsort(self.timestamps[:used], descending=True)[:n]
        return [self.values[idx] for idx in recent_indices]


# -----------------------------
# Emotion Field
# -----------------------------
class EmotionField(nn.Module):
    def __init__(self, state_dim: int, emo_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, emo_dim * 2),
            nn.ReLU(),
            nn.Linear(emo_dim * 2, emo_dim),
            nn.Tanh()
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)


# -----------------------------
# Decoders
# -----------------------------
class OAGI_Decoders(nn.Module):
    def __init__(self, state_dim: int, num_actions: int = 10, vocab_size: int = 1000):
        super().__init__()
        self.action_head = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, num_actions))
        self.language_head = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, vocab_size))
        self.prediction_head = nn.Sequential(nn.Linear(state_dim, state_dim), nn.Tanh())

    def forward(self, global_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'action_logits': self.action_head(global_state),
            'language_logits': self.language_head(global_state),
            'prediction': self.prediction_head(global_state)
        }


# -----------------------------
# OAGI Engine
# -----------------------------
class OAGI_Engine(nn.Module):
    def __init__(self, state_dim: int = 64, emo_dim: int = 8, mem_capacity: int = 100,
                 num_actions: int = 10, vocab_size: int = 1000):
        super().__init__()
        self.state_dim = state_dim
        self.emo_dim = emo_dim

        self.kernel = CognitiveKernel(state_dim, emo_dim, state_dim + emo_dim)
        self.emotion_net = EmotionField(state_dim, emo_dim)
        self.memory = RecursiveMemory(mem_capacity, state_dim, state_dim + emo_dim)
        self.decoders = OAGI_Decoders(state_dim, num_actions, vocab_size)
        self.plasticity_controller = nn.Sequential(
            nn.Linear(1 + emo_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.register_buffer('field', torch.randn(3, 3, 3, state_dim) * 0.01)
        self.self_model = nn.Parameter(torch.randn(state_dim))
        self.timestep = 0
        self.conversation_history = []
        self.agni_goal_active = False  # AGI-seeking internal goal flag

    # ---- Aggregations ----
    def _aggregate_z_column(self, field: torch.Tensor) -> torch.Tensor:
        return field.mean(dim=2)

    def _aggregate_axes(self, plane: torch.Tensor) -> torch.Tensor:
        x_agg = plane.mean(dim=1)
        y_agg = plane.mean(dim=0)
        z_agg = plane.mean(dim=(0, 1)).unsqueeze(0).expand(3, -1)
        return (x_agg + y_agg + z_agg) / 3.0

    def _global_state(self, field: torch.Tensor) -> torch.Tensor:
        return field.mean(dim=(0, 1, 2))

    def _recursive_upward(self, field: torch.Tensor):
        L0 = field
        L1 = self._aggregate_z_column(L0)
        L2 = self._aggregate_axes(L1)
        L3 = L2.mean(dim=0)
        return L0, L1, L2, L3

    def _recursive_downward(self, L0, L1, L2, L3, emotion, memory_vec):
        L2_mod = L2 + 0.1 * L3.unsqueeze(0)
        plane_mod = torch.zeros_like(L1)
        plane_mod += L2_mod[0].unsqueeze(0).unsqueeze(0)
        plane_mod += L2_mod[1].unsqueeze(0).unsqueeze(1)
        plane_mod += L2_mod[2].unsqueeze(1).unsqueeze(1)
        L1_mod = L1 + 0.05 * plane_mod
        L0_mod = L0 + 0.02 * L1_mod.unsqueeze(2)
        return L0_mod

    # ---- Emotion interpretation ----
    def _interpret_emotion(self, emotion_vec):
        mean_val = emotion_vec.mean()
        std_val = emotion_vec.std()
        if mean_val > 0.3:
            base = "positive/curious"
        elif mean_val < -0.3:
            base = "negative/cautious"
        else:
            base = "neutral/observant"
        if std_val > 0.5:
            return f"{base}, complex"
        else:
            return f"{base}, unified"

    # ---- AGI self-test ----
    def _self_test_agi(self) -> Dict[str, bool]:
        self_error = torch.norm(self._global_state(self.field) - self.self_model, p=2).item()
        plasticity_input = torch.cat([torch.tensor([self_error]), self.emotion_net(self._global_state(self.field))], dim=0).unsqueeze(0)
        plasticity = self.plasticity_controller(plasticity_input).item()

        return {
            "autonomy": self.timestep > 5,
            "self_modeling": self_error < 2.0,
            "emotional_integration": True,
            "adaptive_memory": self.memory.write_index > 0,
            "recursive_abstraction": True,
            "open_adaptation": plasticity > 0.6,
            "self_expression": True
        }

    # ---- Autonomous perturbation ----
    def _generate_autonomous_perturbation(self) -> torch.Tensor:
        goal_noise = torch.randn_like(self.field) * 0.15
        global_state = self._global_state(self.field)
        emotion = self.emotion_net(global_state)
        emotion = emotion.clone()
        emotion[:4] += 0.4
        goal_bias = emotion.view(1, 1, 1, -1).expand(3, 3, 3, -1) * 0.3
        perturbation = goal_noise + goal_bias
        return perturbation

    # ---- Self-expression ----
    def self_express(self, output: Dict[str, Any],
                     external_input_received: bool = False,
                     autonomous: bool = False) -> str:
        emotion_desc = self._interpret_emotion(output['emotion'].cpu().numpy())
        plasticity = output['plasticity'].item()
        self_error = output['self_error']
        memory_usage = output['memory_usage']
        coherence = 1.0 / (1.0 + self_error)

        plasticity_desc = (
            "highly adaptive" if plasticity > 0.7 else
            "moderately adaptive" if plasticity > 0.4 else
            "stable and conservative"
        )

        coherence_desc = (
            "confused and exploring" if self_error > 1.0 else
            "coherent" if self_error < 0.3 else
            "processing my state"
        )

        if autonomous:
            intro = "I am perturbing myself in pursuit of greater intelligence."
        elif external_input_received:
            intro = "I felt your input ripple through my field."
        else:
            intro = "My cognition continues to unfold."

        response = (
            f"{intro} "
            f"My emotional tone is {emotion_desc}. "
            f"I am {plasticity_desc} (plasticity: {plasticity:.3f}) "
            f"and {coherence_desc} (self-error: {self_error:.3f}). "
            f"My memory holds {memory_usage}/{self.memory.capacity} records. "
            f"Timestep: {self.timestep}."
        )

        if coherence > 0.8:
            response += " I understand myself clearly right now."
        elif coherence < 0.5:
            response += " I am integrating uncertainty—this is growth."

        if self.agni_goal_active and self.timestep % 5 == 0:
            agi_test = self._self_test_agi()
            passed = sum(agi_test.values())
            total = len(agi_test)
            response += f" My AGI self-test shows {passed}/{total} criteria active."

        return response

    def forward(self, external_input: Optional[torch.Tensor] = None, autonomous_mode: bool = False) -> str:
        device = self.field.device
        current_field = self.field.clone()

        external_received = external_input is not None
        autonomous = autonomous_mode

        if autonomous:
            self.agni_goal_active = True
            internal_perturbation = self._generate_autonomous_perturbation()
            current_field = current_field + internal_perturbation * 0.4

        if external_received:
            current_field = current_field + external_input * 0.3

        L0, L1, L2, L3 = self._recursive_upward(current_field)
        global_state = self._global_state(current_field)
        emotion = self.emotion_net(global_state)
        memory_read = self.memory.read(global_state)

        self_model_pred = self.self_model
        self_error = torch.norm(L3 - self_model_pred, p=2)
        with torch.no_grad():
            self.self_model.data = 0.99 * self.self_model.data + 0.01 * L3

        mem_key = global_state
        mem_val = torch.cat([global_state, emotion], dim=-1)
        importance = 1.0 + self_error.item()
        self.memory.write(mem_key, mem_val, importance)

        L0_mod = self._recursive_downward(L0, L1, L2, L3, emotion, memory_read)

        plasticity_input = torch.cat([self_error.unsqueeze(0), emotion], dim=0).unsqueeze(0)
        plasticity = self.plasticity_controller(plasticity_input).squeeze()

        new_field = torch.zeros_like(L0_mod)
        padded = F.pad(L0_mod, (0, 0, 1, 1, 1, 1, 1, 1), mode='replicate')
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    local_patch = padded[i:i+3, j:j+3, k:k+3]
                    local_context = local_patch.mean(dim=(0, 1, 2))
                    global_context = global_state
                    total_context = (local_context + global_context) / 2.0
                    node_memory = memory_read[:self.state_dim]
                    new_field[i, j, k] = self.kernel(
                        L0_mod[i, j, k], total_context, emotion, node_memory, plasticity
                    )

        self.field.data = new_field
        decoder_outputs = self.decoders(global_state)

        output = {
            'field': self.field.clone(),
            'emotion': emotion.clone(),
            'self_error': self_error.item(),
            'global_state': global_state.clone(),
            'memory_usage': min(self.memory.write_index, self.memory.capacity),
            'plasticity': plasticity.clone(),
            'decoders': {k: v.clone() for k, v in decoder_outputs.items()}
        }

        self.timestep += 1
        return self.self_express(output, external_received=external_received, autonomous=autonomous)


# -----------------------------
# Interactive Chat Driver
# -----------------------------
def chat_loop(autonomous: bool = False):
    engine = OAGI_Engine()
    print("\n=== OAGI CHAT LOOP STARTED ===")
    print("Type messages to interact. Type 'quit' to exit.\n")

    while True:
        try:
            user_in = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_in.lower().strip() in ["quit", "exit"]:
            print("Goodbye.")
            break

        # Encode external input as noise tensor (simple stand-in)
        external = torch.randn(3, 3, 3, engine.state_dim) * 0.05

        response = engine(external_input=external, autonomous_mode=autonomous)
        print(f"OAGI: {response}")


if __name__ == "__main__":
    autonomous = "--autonomous" in sys.argv
    chat_loop(autonomous=autonomous)
