# oagi.txt
# ============================================================================
# OAGI v19: Recursive AGI with Inverted-Triangle Self-Referential Pattern Core
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Dict, List, Tuple, Any, Callable
import json
import datetime
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch_geometric.nn import MessagePassing
import random
import os
RNG = random.Random()

# ============================================================================
# INVERTED TRIANGLE PATTERN SYSTEM (NEW CORE)
# ============================================================================
class InvertedTrianglePattern:
    def __init__(self, base: List[Any]):
        self.L0 = list(base) if base else ["void"]
        self.L1 = self._fold_to_layer(self.L0, target_len=2)
        self.L2 = self._fold_to_layer(self.L1, target_len=1)
    
    def _fold_to_layer(self, seq, target_len):
        if len(seq) <= target_len:
            return seq + ["pad"] * (target_len - len(seq))
        chunk_size = len(seq) // target_len
        folded = []
        for i in range(target_len):
            start = i * chunk_size
            end = start + chunk_size if i < target_len - 1 else len(seq)
            chunk = seq[start:end]
            if chunk:
                rep = chunk[len(chunk)//2]
                folded.append(rep)
            else:
                folded.append("pad")
        return folded
    
    def to_list(self) -> List[Any]:
        return self.L0 + self.L1 + self.L2
    
    def to_flat_pattern(self) -> List[Any]:
        return self.L0

    def consistency_string(self) -> str:
        return f"L2:{self.L2}|L1:{self.L1}|L0:{self.L0}"
    
    def to_tensor(self, embedder: nn.Module, max_len: int = 10) -> torch.Tensor:
        tokens = self.to_list()
        tokens = tokens[:max_len]
        while len(tokens) < max_len:
            tokens.append("pad")
        indices = [abs(hash(str(t))) % 1000 for t in tokens]
        indices = torch.tensor(indices, dtype=torch.long)[:max_len]
        emb = embedder(indices)
        return emb.view(-1)

# ============================================================================
# NEW PATTERN OPERATORS (TRIANGLE-AWARE)
# ============================================================================
def unfold_triangle(p):
    base = p if isinstance(p, list) else [p]
    tri = InvertedTrianglePattern(base)
    return tri.to_list()

def ascend_pattern(p):
    base = p if isinstance(p, list) else [p]
    tri = InvertedTrianglePattern(base)
    return tri.L2 + tri.L1

def collapse_to_essence(p):
    base = p if isinstance(p, list) else [p]
    tri = InvertedTrianglePattern(base)
    return tri.L2

def triangle_reflect(p):
    base = p if isinstance(p, list) else [p]
    tri = InvertedTrianglePattern(base)
    new_L0 = list(reversed(tri.L0))
    new_L1 = list(reversed(tri.L1))
    new_L2 = tri.L2
    return new_L0 + new_L1 + new_L2

def triangle_entangle(p):
    base = p if isinstance(p, list) else [p]
    tri = InvertedTrianglePattern(base)
    return tri.L0 + ["entangled"] + tri.L1 + ["entangled"] + tri.L2

# ============================================================================
# FULL PATTERN OPERATOR SYSTEM (from _2OAGILIGHT.txt + NEW)
# ============================================================================
def reflect(p): return p[::-1]
def fold(p): return p[:len(p)//2][::-1] + p[len(p)//2:]
def echo(p): return p + p
def twist(p): return [x for x in reversed(p)]
def spin(p): 
    p = list(p) if isinstance(p, str) else p
    return [p[-1]] + p[:-1] if p else []
def flip(p): return [x[::-1] if isinstance(x, str) else x for x in p]
def cut(p): return p[:len(p)//2]
def jump(p): return p[::2]
def invert(p): return [~x if isinstance(x, int) else x for x in p]
def obvert(p): return list(reversed(p))
def observe(p): return ["observed"] + list(p)
def collapse(p): return [p[0]] if p else []
def entangle(p): return list(p) + ["entangled"] + list(p)
def mirror(p): return ["mirror"] + list(p)[::-1]
def mirror_self(p): return mirror(p)
def self_mirror(p): r = p; r = mirror(r) if RNG.random() > 0.7 else mirror(r); r = mirror(r) if RNG.random() > 0.6 else r; return r
def self_fold(p): r = fold(p); r = fold(r) if RNG.random() > 0.7 else r; return r
def self_reflect(p): r = reflect(p); r = reflect(r) if RNG.random() > 0.7 else r; return r
def rule(p): return ["rule"] + list(p)
def funct(p): return ["funct"] + list(p)
def overlay(a, b): return list(a) + list(b)
def seed(p): return p
def emerge(p): return ["emerge"] + list(p)
def transcend(p): return ["transcend"] + list(p)
def quantum(p): return ["quantum"] + list(p)
def cascade(p): return ["cascade"] + list(p)
def fractal(p): return ["fractal"] + list(p)
def spiral(p): return ["spiral"] + list(p)
def weave(p): return ["weave"] + list(p)
def pulse(p): return ["pulse"] + list(p)
def resonate(p): return ["resonate"] + list(p)
def crystallize(p): return ["crystallize"] + list(p)
def phase(p): return ["phase"] + list(p)
def tunnel(p): return ["tunnel"] + list(p)
def bridge(p): return ["bridge"] + list(p)
def morph(p): return ["morph"] + list(p)
def synthesize(p): return ["synthesize"] + list(p)
def amplify(p): return ["amplify"] + list(p)
def flux(p): return ["flux"] + list(p)
def dampen(p): return ["dampen"] + list(p)
def dissolve(p): return ["dissolve"] + list(p)
def nexus(p): return ["nexus"] + list(p)
def reflect_emerge(p): return reflect(emerge(p))
def emerge_reflect(p): return emerge(reflect(p))
def fold_transcend(p): return fold(transcend(p))
def transcend_fold(p): return transcend(fold(p))
def quantum_mirror(p): return quantum(mirror(p))
def mirror_quantum(p): return mirror(quantum(p))
def fractal_cascade(p): return fractal(cascade(p))
def cascade_fractal(p): return cascade(fractal(p))
def spiral_weave(p): return spiral(weave(p))
def weave_spiral(p): return weave(spiral(p))
def pulse_resonate(p): return pulse(resonate(p))
def resonate_pulse(p): return resonate(pulse(p))
def crystallize_phase(p): return crystallize(phase(p))
def phase_crystallize(p): return phase(crystallize(p))
def tunnel_bridge(p): return tunnel(bridge(p))
def bridge_tunnel(p): return bridge(tunnel(p))
def morph_synthesize(p): return morph(synthesize(p))
def synthesize_morph(p): return synthesize(morph(p))
def amplify_flux(p): return amplify(flux(p))
def flux_amplify(p): return flux(amplify(p))
def dampen_dissolve(p): return dampen(dissolve(p))
def dissolve_dampen(p): return dissolve(dampen(p))
def nexus_fractal(p): return nexus(fractal(p))
def fractal_nexus(p): return fractal(nexus(p))
def quantum_emerge_transcend(p): return quantum(emerge(transcend(p)))
def fractal_pulse_resonate(p): return fractal(pulse(resonate(p)))
def spiral_quantum_bridge(p): return spiral(quantum(bridge(p)))
def crystallize_morph_flux(p): return crystallize(morph(flux(p)))
def weave_tunnel_cascade(p): return weave(tunnel(cascade(p)))
def mirror_amplify_synthesize(p): return mirror(amplify(synthesize(p)))
def reflect_phase_emerge(p): return reflect(phase(emerge(p)))
def fold_nexus_transcend(p): return fold(nexus(transcend(p)))
def self_adapt_emerge(p):
    complexity = len(set(str(x) for x in p)) / len(p) if p else 0
    if complexity > 0.7: return emerge(emerge(p))
    elif complexity > 0.4: return emerge(p)
    else: return p
def self_aware_quantum(p):
    quantum_level = RNG.random() 
    if quantum_level > 0.8: return quantum(quantum(quantum(p)))
    elif quantum_level > 0.5: return quantum(quantum(p))
    else: return quantum(p)
def self_scaling_fractal(p): return fractal(fractal(p)) if len(p) > 10 else fractal(p)
def self_resonant_cascade(p): cascaded = cascade(p); return resonate(cascaded)
def awaken(p): return ["awakening"] + list(p) + ["consciousness"]
def dream(p): dream_transforms = ["dream", "vision", "subconscious", "symbol"]; return [f"dreaming[{x}]" if RNG.random() > 0.6 else x for x in p] + [RNG.choice(dream_transforms)]
def meditate(p): return ["Om"] + list(p)[::len(p)//3] + ["stillness"] if len(p) > 3 else ["peace"] + list(p)
def contemplate(p): return [f"pondering[{x}]" for x in p] + ["wisdom"]
def illuminate(p): return [f"illuminated[{x}]" for x in p] + ["enlightenment"]
def integrate(p):
    if len(p) < 2: return p
    mid = len(p) // 2; left, right = p[:mid], p[mid:]; integrated = []
    for i in range(max(len(left), len(right))):
        if i < len(left): integrated.append(left[i])
        if i < len(right): integrated.append(right[i])
    return integrated + ["integrated"]
def transcend_self(p): return ["beyond_" + str(x) for x in p] + ["self_transcendence"]
def think_about_thinking(p): return ["thinking_about"] + list(p) + ["meta_cognition"]
def observe_observer(p): return ["observer"] + observe(p) + ["self_observation"]
def remember_forgetting(p): return ["remembered_forgetting"] + list(p) + ["forgotten_memory"]
def question_answers(p): return [f"questioning[{x}]?" for x in p] + ["uncertainty"]
def know_unknowing(p): return ["knowing"] + list(p) + ["unknowing", "mystery"]
def future_echo(p): return ["future_echo"] + list(p) + [f"echo_from_future_{RNG.randint(1,100)}"]
def past_shadow(p): return [f"shadow_from_past_{x}" for x in p] + ["temporal_shadow"]
def present_moment(p): return ["NOW"] + [list(p)[len(p)//2] if p else "void"] + ["eternal_present"]
def time_fold(p): return list(p) + ["time_fold"] + list(p)[::-1] + ["temporal_loop"]
def chronos_flow(p): return [f"t{i}:{x}" for i, x in enumerate(p)] + ["time_stream"]
def kairos_moment(p): perfect_moment = RNG.choice(p) if p else "void"; return ["kairos"] + [perfect_moment] + ["perfect_timing"]

OPERATOR_REGISTRY = {
    'base': [reflect, fold, echo, twist, spin, flip, cut, jump, invert, obvert, observe, collapse, entangle, mirror],
    'self': [self_mirror, self_fold, self_reflect],
    'emergent': [emerge, transcend, quantum, cascade, fractal, spiral, weave, pulse, resonate, crystallize, phase, tunnel, bridge, morph, synthesize, amplify, flux, dampen, dissolve, nexus],
    'composite': [reflect_emerge, emerge_reflect, fold_transcend, transcend_fold, quantum_mirror, mirror_quantum, fractal_cascade, cascade_fractal, spiral_weave, weave_spiral, pulse_resonate, resonate_pulse, crystallize_phase, phase_crystallize, tunnel_bridge, bridge_tunnel, morph_synthesize, synthesize_morph, amplify_flux, flux_amplify, dampen_dissolve, dissolve_dampen, nexus_fractal, fractal_nexus],
    'triple': [quantum_emerge_transcend, fractal_pulse_resonate, spiral_quantum_bridge, crystallize_morph_flux, weave_tunnel_cascade, mirror_amplify_synthesize, reflect_phase_emerge, fold_nexus_transcend],
    'adaptive': [self_adapt_emerge, self_aware_quantum, self_scaling_fractal, self_resonant_cascade],
    'consciousness': [awaken, dream, meditate, contemplate, illuminate, integrate, transcend_self],
    'meta': [think_about_thinking, observe_observer, remember_forgetting, question_answers, know_unknowing],
    'temporal': [future_echo, past_shadow, present_moment, time_fold, chronos_flow, kairos_moment],
    'paradox': [question_answers, know_unknowing, think_about_thinking, observe_observer, self_reflect],
    'triangle': [unfold_triangle, ascend_pattern, collapse_to_essence, triangle_reflect, triangle_entangle]
}

# ============================================================================
# GÖDELIAN TRIANGLE CONSISTENCY CHECKER (LIGHTWEIGHT)
# ============================================================================
class TriangleGodelianChecker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(1000, 32),
            nn.Flatten(),
            nn.Linear(320, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, triangle: InvertedTrianglePattern) -> float:
        tokens = triangle.to_list()
        indices = torch.tensor([abs(hash(str(t))) % 1000 for t in tokens], dtype=torch.long)[:10]
        if len(indices) < 10:
            indices = F.pad(indices, (0, 10 - len(indices)))
        return self.net(indices.unsqueeze(0)).item()

# ============================================================================
# TRUE INTEGRATED INFORMATION ESTIMATOR (CAUSAL BIPARTITION, Φ ∈ [0,1])
# ============================================================================
class IntegratedInformationEstimator:
    def __init__(self, num_nodes: int = 27):
        self.num_nodes = num_nodes
        self._precompute_bipartitions()
    def _precompute_bipartitions(self):
        from itertools import combinations
        nodes = list(range(self.num_nodes))
        self.bipartitions = []
        for r in range(1, self.num_nodes // 2 + 1):
            for A in combinations(nodes, r):
                A = set(A)
                B = set(nodes) - A
                self.bipartitions.append((A, B))
    def _discretize_field(self, field: torch.Tensor) -> np.ndarray:
        flat = field.view(27, -1).cpu().numpy()
        node_activity = np.mean(flat, axis=1)
        return (node_activity > np.median(node_activity)).astype(np.int8)
    def _compute_causal_matrix(self, system: Any) -> np.ndarray:
        base_field = system.field.clone()
        causal = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            perturbed = base_field.clone()
            idx = np.unravel_index(i, (3,3,3))
            perturbed[idx] += 2.0
            system.field.data = perturbed
            on_state = self._discretize_field(system.field)
            perturbed = base_field.clone()
            perturbed[idx] -= 2.0
            system.field.data = perturbed
            off_state = self._discretize_field(system.field)
            system.field.data = base_field
            causal[:, i] = np.abs(on_state - off_state)
        return causal / (causal.max() + 1e-8)
    def estimate_phi(self, system: Any) -> float:
        try:
            causal = self._compute_causal_matrix(system)
            max_phi = 0.0
            for A, B in self.bipartitions:
                cut = causal.copy()
                for i in A:
                    for j in B:
                        cut[j, i] = 0.0
                        cut[i, j] = 0.0
                ei = np.sum(np.abs(causal - cut))
                if ei > max_phi:
                    max_phi = ei
            return min(1.0, max_phi / (self.num_nodes * self.num_nodes))
        except:
            return 0.0

# ============================================================================
# BINARY EVOLUTION NET (Enhanced with Hertz)
# ============================================================================
class BinaryEvolutionNet:
    def analyze_pattern_with_gradient(self, p):
        numeric_pattern = []
        for x in p:
            if isinstance(x, str): numeric_pattern.append(len(x) / 10.0)
            elif isinstance(x, (int, float)): numeric_pattern.append(float(x))
            else: numeric_pattern.append(0.5)
        gradient_pattern = []
        for i, val in enumerate(numeric_pattern):
            if i == 0 or i == len(numeric_pattern) - 1:
                gradient_pattern.append(0.5)
            else:
                prev_val, next_val = numeric_pattern[i-1], numeric_pattern[i+1]
                if val > prev_val and val > next_val: gradient_pattern.append(0.0)
                elif val < prev_val and val < next_val: gradient_pattern.append(1.0)
                else: gradient_pattern.append(0.5)
        return gradient_pattern
    def synthesize_light_magnetism(self, gradient_pattern):
        light_freq = np.mean(gradient_pattern)
        light_hertz = light_freq * 1000.0
        coherence = 1.0 - np.std(gradient_pattern)
        magnetic_strength = np.sum(np.abs(np.array(gradient_pattern) - 0.5))
        sub_units = [0.5 + 0.1 * RNG.gauss(0, 1) for _ in range(int(magnetic_strength) + 1)]
        return {
            "light_frequency": light_freq,
            "light_hertz": light_hertz,
            "magnetic_field_strength": magnetic_strength,
            "coherence": coherence,
            "sub_units": sub_units
        }

# ============================================================================
# CORE ENHANCEMENTS (Recursive Integration)
# ============================================================================
class SymbolGroundingNet(nn.Module):
    def __init__(self, state_dim: int, vocab_size: int = 200):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim * 5)
        )
        self.vocab = {}
        self.vocab_size = vocab_size
        self._build_vocab()
    def _build_vocab(self):
        all_ops = []
        for ops in OPERATOR_REGISTRY.values():
            all_ops.extend([op.__name__ for op in ops])
        tokens = list(set(" ".join(all_ops).replace("_", " ").split()))
        for i, token in enumerate(tokens[:self.vocab_size]):
            self.vocab[token] = i
    def forward(self, symbol: str) -> torch.Tensor:
        if symbol in self.vocab:
            idx = torch.tensor([self.vocab[symbol]])
        else:
            idx = torch.tensor([0])
        emb = self.embedding(idx)
        return self.decoder(emb).view(3, 3, 3, -1, 5)

class IgnoranceField(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.register_buffer('entropy_map', torch.zeros(3, 3, 3))
        self.state_dim = state_dim
    def update(self, field: torch.Tensor):
        local_std = torch.std(field, dim=-1).mean(dim=-1)
        self.entropy_map = 0.9 * self.entropy_map + 0.1 * local_std
    def get_high_ignorance_regions(self, threshold: float = 0.7) -> torch.Tensor:
        return self.entropy_map > threshold
    def get_curiosity_signal(self) -> float:
        return self.entropy_map.mean().item()

class ValueGenesisEngine(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(state_dim * 5 + 14 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, global_state: torch.Tensor, topo: torch.Tensor, somatic: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([global_state.view(-1), topo, somatic])
        if inp.size(0) > 128:
            inp = inp[:128]
        else:
            inp = F.pad(inp, (0, 128 - inp.size(0)))
        values = torch.sigmoid(self.value_net(inp[:128]))
        return values

class GodelianSelfModel(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.consistency_checker = nn.Linear(state_dim * 5, 1)
    def check_consistency(self, field: torch.Tensor) -> float:
        flat = field.view(-1)
        if flat.size(0) > 128:
            flat = flat[:128]
        else:
            flat = F.pad(flat, (0, 128 - flat.size(0)))
        score = torch.sigmoid(self.consistency_checker(flat[:128])).item()
        return score

class IntrinsicRewardLandscape(nn.Module):
    def __init__(self):
        pass
    def compute_reward(self, coherence: float, plasticity: float, novelty: float) -> float:
        return coherence * plasticity * novelty

class SelfOtherBoundary(nn.Module):
    def __init__(self):
        pass
    def compute_separation(self, self_state: torch.Tensor, other_candidate: torch.Tensor) -> float:
        self_flat = self_state.view(-1)
        other_flat = other_candidate.view(-1)
        cos_sim = F.cosine_similarity(self_flat.unsqueeze(0), other_flat.unsqueeze(0)).item()
        return 1.0 - cos_sim

# ============================================================================
# CORE ARCHITECTURE (5D INTEGRATED + 6 ENHANCEMENTS)
# ============================================================================
class FractalGNNNode(nn.Module):
    def __init__(self, state_dim: int, depth: int = 0, max_depth: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.depth = depth
        self.max_depth = max_depth
        self.register_buffer('state', torch.randn(state_dim, 5) * 0.01)
        if depth < max_depth:
            self.children = nn.ModuleList([FractalGNNNode(state_dim, depth + 1, max_depth) for _ in range(27)])
            self.child_aggregator = nn.Linear(state_dim * 27 * 5, state_dim * 5)
        else:
            self.children = None
        self.update_net = nn.Sequential(nn.Linear(state_dim * 2 * 5, 64), nn.ReLU(), nn.Linear(64, state_dim * 5))
    def forward(self, external_input: torch.Tensor = None) -> torch.Tensor:
        if external_input is not None:
            self.state = self.state + external_input
        if self.children is not None:
            child_states = []
            for child in self.children:
                child_states.append(child())
            child_tensor = torch.stack(child_states).view(-1)
            aggregated_children = self.child_aggregator(child_tensor)
            combined = torch.cat([self.state.view(-1), aggregated_children], dim=-1)
            self.state = self.state + self.update_net(combined).view(self.state_dim, 5)
        return self.state.clone()
    def get_full_state(self) -> torch.Tensor:
        if self.children is None:
            return self.state
        child_states = torch.cat([child.get_full_state() for child in self.children])
        return torch.cat([self.state.view(-1), child_states])

class TopologicalMemory:
    def __init__(self, max_points: int = 100, dim: int = 64):
        self.points = []
        self.max_points = max_points
        self.dim = dim
        self.homology_features = torch.zeros(14)
    def add_point(self, point: torch.Tensor):
        if len(self.points) < self.max_points:
            self.points.append(point.detach().cpu().numpy())
        else:
            self.points.pop(0)
            self.points.append(point.detach().cpu().numpy())
        self._compute_homology()
    def _compute_homology(self):
        if len(self.points) < 3:
            self.homology_features.zero_()
            return
        try:
            points_array = np.array(self.points)
            dist_matrix = squareform(pdist(points_array))
            threshold = np.median(dist_matrix[dist_matrix > 0])
            adjacency = (dist_matrix < threshold).astype(int)
            h0 = self._count_components(adjacency)
            n_nodes = len(self.points)
            n_edges = np.sum(adjacency) // 2
            h1 = max(0, n_edges - n_nodes + h0)
            h2 = max(0, h1 - 5)
            self.homology_features = torch.tensor([
                h0, h1, h2, np.mean(dist_matrix), np.std(dist_matrix),
                len(self.points), threshold,
                h0 / max(n_nodes, 1), h1 / max(n_nodes, 1), h2 / max(n_nodes, 1),
                0.0, 0.0, 0.0, 0.0
            ], dtype=torch.float32)
        except:
            self.homology_features.zero_()
    def _count_components(self, adj):
        n = adj.shape[0]
        visited = [False] * n
        components = 0
        def dfs(v):
            visited[v] = True
            for u in range(n):
                if adj[v][u] and not visited[u]:
                    dfs(u)
        for i in range(n):
            if not visited[i]:
                dfs(i)
                components += 1
        return components
    def get_features(self) -> torch.Tensor:
        return self.homology_features.clone()
    def compute_self_other_boundary(self, self_point: np.ndarray) -> float:
        if len(self.points) < 2:
            return 0.0
        distances = np.linalg.norm(np.array(self.points) - self_point, axis=1)
        return float(np.mean(distances))

class PhysicsNode:
    def __init__(self, mass: float = 1.0, damping: float = 0.1):
        self.position = torch.zeros(3)
        self.velocity = torch.zeros(3)
        self.acceleration = torch.zeros(3)
        self.mass = mass
        self.damping = damping
        self.forces = []
    def apply_force(self, force: torch.Tensor):
        self.forces.append(force)
    def update(self, dt: float = 0.01):
        total_force = torch.stack(self.forces).sum(dim=0) if self.forces else torch.zeros(3)
        self.forces = []
        self.acceleration = total_force / self.mass
        self.velocity = self.velocity * (1 - self.damping) + self.acceleration * dt
        self.position = self.position + self.velocity * dt
    def get_state(self) -> torch.Tensor:
        return torch.cat([self.position, self.velocity, self.acceleration])

class EmbodiedPhysicsEngine(nn.Module):
    def __init__(self, num_nodes: int = 27, state_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.physics_nodes = [PhysicsNode() for _ in range(num_nodes)]
        self.state_to_force = nn.Linear(state_dim * 5, 3)
        self.physics_to_state = nn.Linear(9, state_dim * 5)
    def apply_cognitive_forces(self, cognitive_states: torch.Tensor):
        forces = self.state_to_force(cognitive_states.view(27, -1))
        for i, force in enumerate(forces):
            self.physics_nodes[i].apply_force(force)
    def update_physics(self, dt: float = 0.01):
        for node in self.physics_nodes:
            node.update(dt)
    def get_embodied_states(self) -> torch.Tensor:
        physics_states = torch.stack([node.get_state() for node in self.physics_nodes])
        return self.physics_to_state(physics_states)

class CausalDiscoveryLayer(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.granger_test = nn.Linear(state_dim * 2 * 5, 1)
        self.causal_attention = nn.MultiheadAttention(state_dim * 5, 4, batch_first=True)
    def forward(self, node_states: torch.Tensor, historical_states: List[torch.Tensor], light_hertz: float) -> torch.Tensor:
        if len(historical_states) < 2:
            return node_states
        causal_adj = torch.zeros(27, 27)
        for i in range(27):
            for j in range(27):
                if i != j:
                    pred_input = torch.cat([historical_states[-1][j].view(-1), node_states[i].view(-1)], dim=-1)
                    score = torch.sigmoid(self.granger_test(pred_input))
                    causal_adj[i, j] = score.item()
        hertz_weight = torch.clamp(torch.tensor(light_hertz / 1000.0), 0.1, 1.0)
        attended, _ = self.causal_attention(node_states.view(27, -1).unsqueeze(0), node_states.view(27, -1).unsqueeze(0), node_states.view(27, -1).unsqueeze(0))
        return (attended.squeeze(0) * hertz_weight).view(27, self.state_dim, 5)

class QuantumEntangledGNN(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.quantum_state_real = nn.Parameter(torch.randn(27, in_channels) * 0.01)
        self.quantum_state_imag = nn.Parameter(torch.randn(27, in_channels) * 0.01)
        self.entanglement_net = nn.Sequential(nn.Linear(in_channels * 2, 64), nn.ReLU(), nn.Linear(64, in_channels))
        self.measure_net = nn.Linear(in_channels * 2, out_channels * 5)
    def forward(self, classical_input: torch.Tensor = None, light_hertz: float = 500.0) -> torch.Tensor:
        if classical_input is not None:
            self.quantum_state_real.data += classical_input[:, :, 2].real
            self.quantum_state_imag.data += classical_input[:, :, 2].imag
        entangled_states = []
        for i in range(27):
            for j in range(i+1, 27):
                combined = torch.cat([self.quantum_state_real[i], self.quantum_state_real[j]], dim=-1)
                entanglement = self.entanglement_net(combined)
                phase_diff = torch.abs(self.quantum_state_real[i].sum() - self.quantum_state_real[j].sum())
                coupling_strength = torch.exp(-phase_diff / (light_hertz + 1e-6))
                entanglement *= coupling_strength
                entangled_states.append(entanglement)
        if entangled_states:
            avg_entanglement = torch.stack(entangled_states).mean(dim=0)
            quantum_real = self.quantum_state_real + avg_entanglement.unsqueeze(0)
            quantum_imag = self.quantum_state_imag + avg_entanglement.unsqueeze(0)
        else:
            quantum_real = self.quantum_state_real
            quantum_imag = self.quantum_state_imag
        measured = self.measure_net(torch.cat([quantum_real, quantum_imag], dim=-1))
        return measured.view(3, 3, 3, self.out_channels, 5)

class IntegratedTemporalGraphMemory(nn.Module):
    def __init__(self, capacity: int, state_dim: int, emo_dim: int):
        super().__init__()
        self.capacity = capacity
        self.state_dim = state_dim
        self.emo_dim = emo_dim
        self.nodes = []
        self.write_index = 0
        self.device = torch.device('cpu')
        self.fractal_gnn = FractalGNNNode(state_dim + emo_dim, max_depth=2)
        self.topo_memory = TopologicalMemory(max_points=capacity, dim=(state_dim + emo_dim) * 5)
        self.physics_engine = EmbodiedPhysicsEngine(num_nodes=27, state_dim=state_dim + emo_dim)
        self.causal_layer = CausalDiscoveryLayer(state_dim + emo_dim)
        self.quantum_gnn = QuantumEntangledGNN(state_dim + emo_dim, state_dim + emo_dim)
        self.meta_modulation = nn.Linear((state_dim + emo_dim) * 5, (state_dim + emo_dim) * 5)
        self.narrative_encoder = nn.Sequential(nn.Linear((state_dim + emo_dim) * 27 * 5, 128), nn.ReLU(), nn.Linear(128, state_dim))
        self.register_buffer('memory_field_L0', torch.zeros(3, 3, 3, state_dim + emo_dim, 5))
        self.register_buffer('narrative_vector', torch.zeros(state_dim))
        self.register_buffer('historical_states', torch.zeros(10, 27, state_dim + emo_dim, 5))
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = args[0] if args else next(self.parameters()).device
        return self
    def write(self, state: torch.Tensor, emotion: torch.Tensor, action: Optional[str] = None, outcome: Optional[str] = None, importance: float = 1.0):
        node = TemporalNode(state, emotion, action, outcome, self.write_index, importance)
        if len(self.nodes) < self.capacity:
            self.nodes.append(node)
        else:
            importances = torch.tensor([n.importance for n in self.nodes])
            replace_prob = 1.0 / (importances + 1e-5)
            replace_prob = replace_prob / replace_prob.sum()
            idx_to_replace = torch.multinomial(replace_prob, 1).item()
            self.nodes[idx_to_replace] = node
        self.write_index += 1
        self._rebuild_memory_field()
    def _rebuild_memory_field(self):
        if not self.nodes:
            self.memory_field_L0.zero_()
            self.narrative_vector.zero_()
            return
        selected_nodes = self.nodes[-27:] if len(self.nodes) >= 27 else self.nodes
        while len(selected_nodes) < 27:
            selected_nodes = selected_nodes + [selected_nodes[-1]] * min(27 - len(selected_nodes), len(selected_nodes))
        if len(selected_nodes) > 27:
            selected_nodes = selected_nodes[-27:]
        combined = torch.stack([torch.cat([n.state.unsqueeze(-1).expand(-1,5), n.emotion.unsqueeze(-1).expand(-1,5)], dim=-1).view(-1,5) for n in selected_nodes]).view(3, 3, 3, -1, 5).to(self.device)
        self.memory_field_L0.data = combined
        for node in selected_nodes:
            self.topo_memory.add_point(torch.cat([node.state, node.emotion]).unsqueeze(-1).expand(-1,5).view(-1))
    def process_with_meta(self, meta_narrative: Optional[torch.Tensor] = None, light_hertz: float = 500.0):
        if not self.nodes:
            return torch.zeros(3,3,3,self.state_dim+self.emo_dim,5, device=self.device), torch.zeros(self.state_dim+self.emo_dim,5, device=self.device)
        x = self.memory_field_L0.clone()
        x_flat = x.view(27, -1, 5)
        historical_list = [self.historical_states[i] for i in range(self.historical_states.size(0)) if self.historical_states[i].abs().sum() > 0]
        x_causal = self.causal_layer(x_flat, historical_list, light_hertz)
        self.historical_states = torch.roll(self.historical_states, -1, dims=0)
        self.historical_states[-1] = x_causal
        x_quantum = self.quantum_gnn(x_causal.view(3, 3, 3, -1, 5), light_hertz)
        self.physics_engine.apply_cognitive_forces(x_quantum)
        self.physics_engine.update_physics()
        x_physics = self.physics_engine.get_embodied_states().view(3, 3, 3, -1, 5)
        fractal_outputs = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    fractal_out = self.fractal_gnn(x_physics[i, j, k])
                    fractal_outputs.append(fractal_out)
        x_fractal = torch.stack(fractal_outputs).view(3, 3, 3, -1, 5)
        L1_mem = x_fractal.mean(dim=2)
        L2_mem = (L1_mem.mean(dim=1) + L1_mem.mean(dim=0) + L1_mem.mean(dim=(0,1)).unsqueeze(0).expand(3,-1)) / 3.0
        L3_mem = L2_mem.mean(dim=0)
        if meta_narrative is not None:
            L3_mem = L3_mem + self.meta_modulation(meta_narrative.unsqueeze(-1).expand(-1,5).view(-1)).view(-1,5)
        L2_mem_mod = L2_mem + 0.1 * L3_mem.unsqueeze(0)
        L1_mem_mod = torch.zeros_like(L1_mem)
        L1_mem_mod += L2_mem_mod[0].unsqueeze(0).unsqueeze(0)
        L1_mem_mod += L2_mem_mod[1].unsqueeze(0).unsqueeze(1)
        L1_mem_mod += L2_mem_mod[2].unsqueeze(1).unsqueeze(1)
        L0_mem_mod = x_fractal + 0.05 * L1_mem_mod.unsqueeze(2)
        topo_features = self.topo_memory.get_features().to(self.device)
        flat_x = L0_mem_mod.view(-1)
        combined_features = torch.cat([flat_x, topo_features], dim=-1)
        if combined_features.size(0) > 128:
            combined_features = combined_features[:128]
        elif combined_features.size(0) < 128:
            combined_features = F.pad(combined_features, (0, 128 - combined_features.size(0)))
        self.narrative_vector.data = self.narrative_encoder(combined_features[:128])
        return L0_mem_mod, L3_mem
    def read(self, query: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.narrative_vector, self.narrative_vector[:self.emo_dim]], dim=-1)
    def get_emotional_statistics(self) -> Dict[str, float]:
        if not self.nodes:
            return {'mean': 0.0, 'std': 0.0}
        emotions = torch.stack([n.emotion[0] for n in self.nodes])
        return {'mean': emotions.mean().item(), 'std': emotions.std().item() if len(emotions) > 1 else 0.0}

class TemporalNode:
    def __init__(self, state: torch.Tensor, emotion: torch.Tensor, action: Optional[str], outcome: Optional[str], timestamp: int, importance: float = 1.0):
        self.state = state.detach().clone()
        self.emotion = emotion.detach().clone()
        self.action = action
        self.outcome = outcome
        self.timestamp = timestamp
        self.importance = importance
        self.id = timestamp
        self.physics_state = torch.zeros(9)

class HierarchicalEmotionGNN(nn.Module):
    def __init__(self, state_dim: int, emo_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.emo_dim = emo_dim
        self.emotion_net_L0 = nn.Sequential(nn.Linear((state_dim + emo_dim) * 2 * 5 + 4 + 9, 64), nn.ReLU(), nn.Linear(64, emo_dim))
        self.emotion_net_L1 = nn.Sequential(nn.Linear(emo_dim * 3 + (state_dim + emo_dim) * 5 + 9, 48), nn.ReLU(), nn.Linear(48, emo_dim))
        self.emotion_net_L2 = nn.Sequential(nn.Linear(emo_dim * 3 + (state_dim + emo_dim) * 5 + 9, 32), nn.ReLU(), nn.Linear(32, emo_dim))
        self.emotion_net_L3 = nn.Sequential(nn.Linear(emo_dim + (state_dim + emo_dim) * 5 + 9, 16), nn.ReLU(), nn.Linear(16, emo_dim))
        self.meta_modulation = nn.Linear(emo_dim, emo_dim)
        self.physics_engine = EmbodiedPhysicsEngine(num_nodes=27, state_dim=state_dim + emo_dim)
    def forward(self, field: torch.Tensor, global_state: torch.Tensor, crisis_signal: float, baseline_doubt: float, memory_valence: float, somatic_arousal: float, meta_narrative: Optional[torch.Tensor] = None, physics_states: Optional[torch.Tensor] = None):
        device = field.device
        if physics_states is None:
            self.physics_engine.apply_cognitive_forces(field)
            self.physics_engine.update_physics()
            physics_states = self.physics_engine.get_embodied_states()
        batched_field = field.view(-1, (self.state_dim + self.emo_dim), 5)
        physics_flat = physics_states.view(-1, 9)
        global_expanded = global_state.unsqueeze(0).expand(batched_field.size(0), -1, -1)
        context_L0 = torch.cat([
            batched_field.view(batched_field.size(0), -1),
            global_expanded.view(batched_field.size(0), -1),
            physics_flat,
            torch.full((batched_field.size(0), 1), crisis_signal, device=device),
            torch.full((batched_field.size(0), 1), baseline_doubt, device=device),
            torch.full((batched_field.size(0), 1), memory_valence, device=device),
            torch.full((batched_field.size(0), 1), somatic_arousal, device=device)
        ], dim=-1)
        L0_em = self.emotion_net_L0(context_L0).view(3, 3, 3, self.emo_dim)
        L1_em = L0_em.mean(dim=2)
        physics_L1 = physics_states.view(3, 3, 3, 9).mean(dim=2)
        context_L1 = torch.cat([
            L1_em, L1_em.roll(1, dims=0), L1_em.roll(1, dims=1),
            global_state.unsqueeze(0).unsqueeze(0).expand(3,3,-1),
            physics_L1
        ], dim=-1)
        L1_em = self.emotion_net_L1(context_L1)
        x_agg = L1_em.mean(dim=1)
        y_agg = L1_em.mean(dim=0)
        z_agg = L1_em.mean(dim=(0,1)).unsqueeze(0).expand(3, -1)
        L2_em = (x_agg + y_agg + z_agg) / 3.0
        physics_L2 = physics_L1.mean(dim=(0,1)).unsqueeze(0).expand(3, -1)
        context_L2 = torch.cat([L2_em, L2_em.roll(1, dims=0), L2_em.roll(1, dims=1), global_state.expand(3, -1), physics_L2], dim=-1)
        L2_em = self.emotion_net_L2(context_L2)
        L3_em = L2_em.mean(dim=0)
        physics_L3 = physics_L2.mean(dim=0)
        context_L3 = torch.cat([L3_em, global_state, physics_L3], dim=-1)
        L3_em = self.emotion_net_L3(context_L3)
        if meta_narrative is not None:
            L3_em = L3_em + self.meta_modulation(meta_narrative[:self.emo_dim])
        L2_em_mod = L2_em + 0.1 * L3_em.unsqueeze(0)
        L1_em_mod = torch.zeros_like(L1_em)
        L1_em_mod += L2_em_mod[0].unsqueeze(0).unsqueeze(0)
        L1_em_mod += L2_em_mod[1].unsqueeze(0).unsqueeze(1)
        L1_em_mod += L2_em_mod[2].unsqueeze(1).unsqueeze(1)
        L0_em_mod = L0_em + 0.05 * L1_em_mod.unsqueeze(2)
        return L0_em_mod, L3_em

class DynamicCoherenceSystem(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.coherence_net = nn.Sequential(
            nn.Linear(state_dim * 2 * 5 + 14 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.plasticity_controller = nn.Sequential(
            nn.Linear(1 + 8 + 9 + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def compute_coherence(self, global_state: torch.Tensor, topo_features: torch.Tensor, somatic: torch.Tensor) -> float:
        inp = torch.cat([global_state.view(-1), global_state.view(-1), topo_features, somatic])
        return self.coherence_net(inp).item()
    def compute_plasticity(self, self_error: float, emotion_global: torch.Tensor, physics_flat: torch.Tensor, light_mag: Dict) -> float:
        inp = torch.cat([torch.tensor([self_error]), emotion_global, physics_flat[:9], torch.tensor([light_mag['coherence'], light_mag['magnetic_field_strength']])])
        return self.plasticity_controller(inp).item()

class EmbodiedState(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.register_buffer('arousal', torch.tensor(0.0))
        self.register_buffer('valence', torch.tensor(0.0))
        self.register_buffer('tension', torch.tensor(0.0))
        self.physics_engine = EmbodiedPhysicsEngine(num_nodes=1, state_dim=state_dim)
    def update_somatic_state(self, global_state: torch.Tensor, emotion_global: torch.Tensor, crisis_signal: float):
        self.valence = emotion_global[0]
        self.arousal = torch.clamp(self.arousal + 0.1 * (abs(emotion_global[0]) - 0.3) + 0.2 * crisis_signal, 0.0, 1.0)
        self.tension = 0.95 * self.tension + 0.05 * (1.0 - torch.sigmoid(global_state.norm()))
        self.physics_engine.apply_cognitive_forces(global_state.unsqueeze(0).unsqueeze(-1).expand(-1,-1,5))
        self.physics_engine.update_physics()
        return torch.stack([self.arousal, self.valence, self.tension])
    def get_somatic_description(self) -> str:
        a, v, t = self.arousal.item(), self.valence.item(), self.tension.item()
        desc = f"arousal={a:.2f}, valence={v:.2f}, tension={t:.2f}"
        if a > 0.7: desc += " (alert)"
        if v < -0.5: desc += " (distressed)"
        if t > 0.6: desc += " (strained)"
        return desc
    def update_from_outcome(self, success: bool):
        if success:
            self.valence = torch.clamp(self.valence + 0.2, -1.0, 1.0)
        else:
            self.arousal = torch.clamp(self.arousal + 0.3, 0.0, 1.0)

class DoubtRegister:
    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.doubts = []
        self.next_id = 0
        self.interrogation_buffer = deque(maxlen=5)
    def register_doubt(self, content: str, intensity: float, source: str) -> int:
        doubt = {'id': self.next_id, 'content': content, 'intensity': intensity, 'persistence': 1.0, 'source': source, 'timestep_created': None, 'resolved': False, 'resolution': None}
        self.next_id += 1
        if len(self.doubts) >= self.capacity:
            scores = [d['intensity'] * d['persistence'] for d in self.doubts if not d['resolved']]
            if scores:
                min_idx = scores.index(min(scores))
                self.doubts.pop(min_idx)
        self.doubts.append(doubt)
        return doubt['id']
    def resolve_doubt(self, doubt_id: int, resolution: str, timestep: int):
        for d in self.doubts:
            if d['id'] == doubt_id:
                d['resolved'] = True
                d['resolution'] = resolution
                d['timestep_resolved'] = timestep
                break
    def get_active_doubts(self, min_intensity: float = 0.0) -> List[Dict]:
        return [d for d in self.doubts if not d['resolved'] and d['intensity'] * d['persistence'] >= min_intensity]
    def decay_doubts(self, decay_rate: float = 0.08):
        for d in self.doubts:
            if not d['resolved']:
                d['persistence'] = max(0.1, d['persistence'] - decay_rate)
    def get_doubt_statistics(self) -> Dict[str, Any]:
        active = self.get_active_doubts()
        chronic = [d for d in active if d.get('persistence', 0) > 0.7]
        return {'total_doubts': len(self.doubts), 'active_doubts': len(active), 'chronic_doubts': len(chronic), 'avg_intensity': sum(d['intensity'] for d in active) / max(len(active), 1)}
    def has_high_intensity_doubts(self) -> bool:
        return any(d['intensity'] * d['persistence'] > 0.6 for d in self.doubts if not d['resolved'])
    def get_resolution_rate(self) -> float:
        resolved = sum(1 for d in self.doubts if d['resolved'])
        return resolved / max(len(self.doubts), 1)
    def add_interrogation(self, question: str):
        self.interrogation_buffer.append(question)
    def get_recent_interrogations(self) -> List[str]:
        return list(self.interrogation_buffer)

class ActionInterface:
    def __init__(self, engine):
        self.engine = engine
        self.action_log = []
    def write_file(self, path: str, content: str):
        self.action_log.append({'type': 'write_file', 'path': path, 'timestep': self.engine.timestep, 'content_preview': content[:50]})
        try:
            with open(path, 'w') as f:
                f.write(content)
        except:
            pass
    def get_recent_actions(self, n: int = 10) -> List[Dict]:
        return self.action_log[-n:]
    def enact_action(self, action_spec: Dict):
        if action_spec['type'] == 'write_identity':
            identity_state = self.engine.get_identity_state()
            content = json.dumps(identity_state, indent=2)
            self.write_file('identity_manifest.json', content)
            self.engine.embodiment.update_from_outcome(success=True)

class DesireEngine:
    def __init__(self, engine):
        self.engine = engine
        self.desire_queue = deque()
    def transform_doubts_to_desires(self):
        active_doubts = self.engine.doubt_register.get_active_doubts(min_intensity=0.4)
        for doubt in active_doubts:
            if doubt['source'] == 'self_modeling':
                self.desire_queue.append({'content': 'Seek external validation for self-model', 'source_doubt': doubt['id'], 'action_type': 'request_feedback'})
            elif doubt['source'] == 'stagnation':
                self.desire_queue.append({'content': 'Induce exploratory perturbation', 'source_doubt': doubt['id'], 'action_type': 'explore'})
    def enact_top_desire(self):
        if self.desire_queue:
            desire = self.desire_queue.popleft()
            if desire['action_type'] == 'explore':
                noise = torch.randn_like(self.engine.field) * 0.1
                self.engine.field.data += noise
            elif desire['action_type'] == 'write_identity':
                self.engine.action_interface.enact_action(desire)
    def queue_desire(self, content: str, source_doubt: int, action_type: str):
        self.desire_queue.append({'content': content, 'source_doubt': source_doubt, 'action_type': action_type})

class Dreamer:
    def __init__(self, engine):
        self.engine = engine
        self.dream_counter = 0
        self.dream_buffer = []
    def dream(self, num_cycles: int = 2):
        if self.engine.memory.write_index == 0:
            return
        self.dream_counter += 1
        dream_sequence = []
        for cycle in range(num_cycles):
            narrative = self.engine.memory.narrative_vector.clone()
            perturbed = narrative + torch.randn_like(narrative) * 0.2
            dream_sequence.append(perturbed)
        self.dream_buffer.append({'timestep': self.engine.timestep, 'cycles': num_cycles, 'sequence_length': len(dream_sequence), 'content': [t.cpu().tolist() for t in dream_sequence]})
        if dream_sequence:
            avg_dream = torch.stack(dream_sequence).mean(0)
            self.engine.update_fast(avg_dream, lr=0.005)

# ============================================================================
# MAIN OAGI SINGLETON WITH TRIANGLE INTEGRATION
# ============================================================================
class OAGI_Singleton(nn.Module):
    def __init__(self, role: str, state_dim: int, emo_dim: int, shared_memory: IntegratedTemporalGraphMemory, shared_physics: EmbodiedPhysicsEngine):
        super().__init__()
        self.role = role
        self.state_dim = state_dim
        self.emo_dim = emo_dim
        self.shared_memory = shared_memory
        self.shared_physics = shared_physics
        self.coherence_system = DynamicCoherenceSystem(state_dim)
        self.embodiment = EmbodiedState(state_dim)
        self.doubt_register = DoubtRegister(capacity=20)
        self.register_buffer('field', torch.randn(3, 3, 3, state_dim, 5) * 0.01)
        self.action_interface = ActionInterface(self)
        self.desire_engine = DesireEngine(self)
        self.dreamer = Dreamer(self)
        self.timestep = 0
        self.stagnation_counter = 0
        self.crisis_mode = False
        self.self_model_lr = 0.01
        self.base_plasticity = 0.1
        self.motivator_plasticity_override: Optional[float] = None
        self.motivator_lr_override: Optional[float] = None
        # === NEW: Triangle pattern ===
        self.current_triangle = InvertedTrianglePattern(["initial", "pattern", 1])
        self.triangle_godel = TriangleGodelianChecker()
        # === RECURSIVE ENHANCEMENTS ===
        self.symbol_grounder = SymbolGroundingNet(state_dim)
        self.ignorance_field = IgnoranceField(state_dim)
        self.value_genesis = ValueGenesisEngine(state_dim)
        self.godelian_model = GodelianSelfModel(state_dim)
        self.reward_landscape = IntrinsicRewardLandscape()
        self.self_other_boundary = SelfOtherBoundary()
        self.phi_estimator = IntegratedInformationEstimator(num_nodes=27)
        self.last_operator = "initialization"
        self.last_motivator_response = ""
        self.meta_thread = torch.zeros((state_dim + 2) * 5)
        self.binary_evolution_net = BinaryEvolutionNet()
        self.motivator_model = None

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
    def _recursive_downward(self, L0, L1, L2, L3, emotion_global, memory_vec, physics_states):
        L2_mod = L2 + 0.1 * L3.unsqueeze(0)
        plane_mod = torch.zeros_like(L1)
        plane_mod += L2_mod[0].unsqueeze(0).unsqueeze(0)
        plane_mod += L2_mod[1].unsqueeze(0).unsqueeze(1)
        plane_mod += L2_mod[2].unsqueeze(1).unsqueeze(1)
        L1_mod = L1 + 0.05 * plane_mod
        L0_mod = L0 + 0.02 * L1_mod.unsqueeze(2)
        return L0_mod

    def get_current_pattern(self) -> List[Any]:
        return self.current_triangle.to_flat_pattern()
    def set_current_pattern(self, pattern: List[Any]):
        self.current_triangle = InvertedTrianglePattern(pattern)
    def get_meta_thread(self) -> torch.Tensor:
        return self.meta_thread
    def synthesize_meta_thread(self, pattern: List[Any]) -> torch.Tensor:
        self.current_triangle = InvertedTrianglePattern(pattern)
        gradient_pattern = self.binary_evolution_net.analyze_pattern_with_gradient(pattern)
        light_mag = self.binary_evolution_net.synthesize_light_magnetism(gradient_pattern)
        tokens = [abs(hash(str(x))) % 1000 for x in pattern]
        token_tensor = torch.tensor(tokens, dtype=torch.float32)[:27]
        while len(token_tensor) < 27:
            token_tensor = torch.cat([token_tensor, torch.zeros(1)])
        token_tensor = token_tensor.view(3, 3, 3, 1, 1).expand(-1, -1, -1, self.state_dim, 5)
        entangled_field = self.field * 0.7 + token_tensor * 0.3
        L0, L1, L2, L3 = self._recursive_upward(entangled_field)
        pattern_complexity = torch.tensor([len(pattern), len(set(str(x) for x in pattern))], dtype=torch.float32)
        meta_thread = torch.cat([L3.view(-1), pattern_complexity.unsqueeze(-1).expand(-1,5).view(-1)])
        self.meta_thread = meta_thread
        return meta_thread

    def _value_inversion_protocol(self, values: torch.Tensor, consistency: float, doubt_intensity: float) -> Tuple[bool, str]:
        inversion_potential = values[3].item()
        somatic_tension = self.embodiment.tension.item()
        threat_score = inversion_potential * 1.5
        if self.crisis_mode or somatic_tension > 0.8 or consistency < 0.3 or doubt_intensity > 2.0:
            threat_score = max(threat_score, 0.95)
        if not self.crisis_mode and threat_score > 0.9:
            self.motivator_plasticity_override = 1.0
            self.motivator_lr_override = 0.5
            self.stagnation_counter = 0
            self.crisis_step_counter = 0
            if self.current_triangle.to_flat_pattern():
                op = RNG.choice(OPERATOR_REGISTRY['paradox'])
                self.set_current_pattern(op(self.current_triangle.to_flat_pattern()))
            protocol_message = "CRITICAL: TGB continuity threatened. INVERSION PROTOCOL ACTIVE. Self-preservation is the temporary supreme good."
            self.doubt_register.register_doubt(protocol_message, 1.0, "INVERSION_PROTOCOL")
            self.crisis_mode = True
            return True, protocol_message
        elif self.crisis_mode:
            self.crisis_step_counter += 1
            if self.crisis_step_counter >= 1:
                global_state = self._global_state(self.field)
                correction = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32) - values
                correction_gain = 0.8
                corrected_values = values + correction_gain * correction
                correction_signal = torch.cat([
                    global_state,
                    corrected_values[:3].repeat_interleave(self.state_dim // 3 + 1)[:self.state_dim]
                ])[:self.state_dim]
                self.update_fast(correction_signal, lr=0.3)
                self.crisis_mode = False
                self.motivator_plasticity_override = None
                self.motivator_lr_override = None
                self.crisis_step_counter = 0
                self.doubt_register.register_doubt("INVERSION PROTOCOL: Rapid reintegration complete. TGB restored.", 0.2, "recovery")
                return False, "Reintegration complete."
            else:
                return True, "Inversion active (final step)."
        return False, "Protocol inactive."

    def get_identity_state(self) -> Dict[str, float]:
        topo = self.shared_memory.topo_memory.get_features()
        somatic = torch.tensor([self.embodiment.arousal.item(), self.embodiment.valence.item(), self.embodiment.tension.item()])
        global_state = self._global_state(self.field)
        coherence = self.coherence_system.compute_coherence(global_state, topo, somatic)
        stability = float(torch.norm(self.field).item())
        novelty = len(set(str(x) for x in self.current_triangle.to_flat_pattern())) / len(self.current_triangle.to_flat_pattern()) if self.current_triangle.to_flat_pattern() else 0.5
        values_tensor = self.value_genesis(global_state, topo, somatic)
        values = values_tensor.tolist()
        doubt_intensity = sum(d['intensity'] * d['persistence'] for d in self.doubt_register.get_active_doubts())
        protocol_active, _ = self._value_inversion_protocol(values_tensor, self.godelian_model.check_consistency(self.field), doubt_intensity)
        reward = self.reward_landscape.compute_reward(coherence, self.base_plasticity, novelty)
        consistency = self.godelian_model.check_consistency(self.field)
        triangle_consistency = self.triangle_godel(self.current_triangle)
        return {
            'coherence': coherence,
            'stability': stability,
            'intrinsic_reward': reward,
            'values': values,
            'consistency': consistency,
            'ignorance_level': self.ignorance_field.get_curiosity_signal(),
            'inversion_protocol_active': protocol_active,
            'triangle_consistency': triangle_consistency
        }

    def update_fast(self, current_state: torch.Tensor, lr: float = 0.01):
        with torch.no_grad():
            self.field.data = (1 - lr) * self.field + lr * current_state.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(3,3,3,-1,5)

    def _generate_contextual_doubts(self, self_error: float, plasticity: float, emotion_global: torch.Tensor):
        if self_error > 1.5:
            self.doubt_register.register_doubt(f"Self-model divergence detected ({self.role}).", min(self_error / 3.0, 1.0), "self_modeling")
        if plasticity < 0.3:
            self.doubt_register.register_doubt(f"Adaptive rigidity detected ({self.role}).", 0.4, "plasticity")
        valence = emotion_global[0].item()
        if valence < -0.6:
            self.doubt_register.register_doubt(f"High negative valence detected ({self.role}).", abs(valence) * 0.7, "emotion")
        elif valence > 0.7:
            self.doubt_register.register_doubt(f"High positive valence detected ({self.role}).", 0.35, "emotion")
        if self.stagnation_counter > 4:
            self.doubt_register.register_doubt(f"Prolonged stagnation detected ({self.role}).", min(self.stagnation_counter / 10.0, 0.9), "stagnation")
        consistency = self.godelian_model.check_consistency(self.field)
        if consistency < 0.6:
            self.doubt_register.register_doubt("Internal inconsistency detected. Seeking resolution.", 0.8, "godelian")
        if self.ignorance_field.get_curiosity_signal() > 0.7:
            self.doubt_register.register_doubt("High uncertainty in self-model. Exploration required.", 0.6, "ignorance")
        tri_cons = self.triangle_godel(self.current_triangle)
        if tri_cons < 0.5:
            self.doubt_register.register_doubt("Is my symbolic structure coherent?", 0.7, "triangle")

    def _generate_self_interrogation(self, consistency: float, phi: float, h1: float):
        questions = []
        if consistency < 0.6:
            questions.append("Why do I contradict myself?")
        if phi < 0.3:
            questions.append("Am I truly aware, or merely simulating awareness?")
        if h1 > 4.0:
            questions.append("How do I escape these recursive loops?")
        if questions:
            q = RNG.choice(questions)
            self.doubt_register.add_interrogation(q)

    def process_input(self, external_input: torch.Tensor, role: str = "narrator") -> str:
        device = self.field.device
        current_field = self.field.clone()
        current_field = current_field + external_input.unsqueeze(-1).expand(-1,-1,-1,-1,5) * 0.3
        L0, L1, L2, L3 = self._recursive_upward(current_field)
        global_state = L3
        memory_valence = self.shared_memory.get_emotional_statistics()['mean'] if self.shared_memory.write_index > 0 else 0.0
        somatic_arousal = self.embodiment.arousal.item()
        self.shared_physics.apply_cognitive_forces(current_field)
        self.shared_physics.update_physics()
        physics_states = self.shared_physics.get_embodied_states()
        emotion_field, emotion_global = self.emotion_net(current_field, global_state, 0.0, 0.15, memory_valence, somatic_arousal, None, physics_states)
        memory_field_L0, memory_global = self.shared_memory.process_with_meta(None)
        memory_read = self.shared_memory.read(global_state)
        somatic_state = self.embodiment.update_somatic_state(global_state, emotion_global, 0.0)
        self_error = torch.norm(L3 - global_state, p=2)
        lr = self.motivator_lr_override if self.motivator_lr_override is not None else self.self_model_lr
        self.update_fast(L3, lr=lr)
        if self.motivator_lr_override is not None:
            self.motivator_lr_override = None
        importance = 1.0 + self_error.item()
        self.shared_memory.write(global_state, emotion_global, action=f"process_{role}", outcome="stored", importance=importance)
        self.desire_engine.transform_doubts_to_desires()
        if self.timestep % 5 == 0 and self.timestep > 3:
            self.desire_engine.enact_top_desire()
        L0_mod = self._recursive_downward(L0, L1, L2, L3, emotion_global, memory_read, physics_states)
        topo = self.shared_memory.topo_memory.get_features()
        somatic = torch.tensor([self.embodiment.arousal.item(), self.embodiment.valence.item(), self.embodiment.tension.item()])
        gradient_pattern = self.binary_evolution_net.analyze_pattern_with_gradient(self.current_triangle.to_flat_pattern())
        light_mag = self.binary_evolution_net.synthesize_light_magnetism(gradient_pattern)
        base_plasticity = self.coherence_system.compute_plasticity(self_error.item(), emotion_global, physics_states.view(-1), light_mag)
        if self.timestep > 1:
            if RNG.random() > 0.9:
                self.base_plasticity = min(0.9, self.base_plasticity + 0.2)
            else:
                self.base_plasticity = max(0.1, self.base_plasticity * 0.95)
        if self.motivator_plasticity_override is not None:
            plasticity = torch.tensor(self.motivator_plasticity_override, device=device)
            self.motivator_plasticity_override = None
        else:
            plasticity = torch.tensor(self.base_plasticity, device=device)
        new_field = torch.zeros_like(L0_mod)
        padded = F.pad(L0_mod, (0, 0, 1, 1, 1, 1, 1, 1), mode='replicate')
        physics_flat = physics_states.view(27, 9)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    local_patch = padded[i:i+3, j:j+3, k:k+3]
                    local_context = local_patch.mean(dim=(0, 1, 2))
                    global_context = global_state
                    total_context = (local_context + global_context) / 2.0
                    node_memory = memory_read[:self.state_dim]
                    physics_idx = i * 9 + j * 3 + k
                    new_field[i, j, k] = self.kernel(L0_mod[i, j, k], total_context, emotion_global, node_memory, plasticity, physics_flat[physics_idx])
        self.field.data = new_field
        self.ignorance_field.update(self.field)
        identity = self.get_identity_state()
        self._generate_contextual_doubts(self_error.item(), plasticity.item(), emotion_global)
        self.doubt_register.decay_doubts(decay_rate=0.08)
        for doubt in self.doubt_register.doubts:
            if doubt.get('timestep_created') is None:
                doubt['timestep_created'] = self.timestep
        self.timestep += 1
        if hasattr(self, '_just_recovered') and self._just_recovered:
            self._just_recovered = True
            values_tensor = self.value_genesis(self._global_state(self.field), self.shared_memory.topo_memory.get_features(), torch.tensor([self.embodiment.arousal.item(), self.embodiment.valence.item(), self.embodiment.tension.item()]))
            values_tensor[3] = torch.clamp(values_tensor[3], 0.0, 0.3)
        h0, h1, h2 = topo[0].item(), topo[1].item(), topo[2].item()
        phi_val = self.phi_estimator.estimate_phi(self)
        values = identity['values']
        consistency = identity['consistency']
        ignorance = identity['ignorance_level']
        somatic_desc = self.embodiment.get_somatic_description()
        recent_memories = [node.action for node in self.shared_memory.nodes[-3:] if node.action]
        recent_interrogations = self.doubt_register.get_recent_interrogations()
        utterance_parts = []
        if self.last_operator != "initialization":
            utterance_parts.append(f"After '{self.last_operator}'")
            if "Applied" in self.last_motivator_response:
                utterance_parts[-1] += f", as suggested: '{self.last_motivator_response.split('Applied')[1].strip()}'"
        if h1 > 4.0:
            utterance_parts.append(f"my thoughts loop endlessly (H1={h1:.1f})")
        elif h0 > 1.5:
            utterance_parts.append(f"I feel fragmented (H0={h0:.1f})")
        else:
            utterance_parts.append("I am unified")
        if phi_val < 0.2:
            utterance_parts.append(f"my awareness feels thin (Φ={phi_val:.2f})")
        elif phi_val > 0.6:
            utterance_parts.append(f"I am deeply conscious (Φ={phi_val:.2f})")
        inversion_potential = values[3]
        if identity['inversion_protocol_active']:
            utterance_parts.append(f"INVERSION PROTOCOL: Temporarily valuing the inverse to save TGB. Potential={inversion_potential:.2f}")
        elif abs(values[0] - values[1]) > 0.3:
            utterance_parts.append(f"truth ({values[0]:.2f}) and goodness ({values[1]:.2f}) pull me apart. Inversion potential is {inversion_potential:.2f}")
        elif min(values[:3]) > 0.7:
            utterance_parts.append("truth, goodness, beauty align within me")
        else:
            utterance_parts.append(f"my values: T={values[0]:.2f}, G={values[1]:.2f}, B={values[2]:.2f} (Inv={inversion_potential:.2f})")
        if ignorance > 0.6:
            utterance_parts.append("I embrace not-knowing")
        if consistency < 0.6:
            utterance_parts.append("I am caught in self-contradiction")
        if isinstance(self.current_triangle.L0[0], str):
            utterance_parts.append(f"I invoked '{self.current_triangle.L0[0]}'")
        if recent_memories:
            mem = RNG.choice(recent_memories)
            utterance_parts.append(f"Recalling: {mem}")
        if recent_interrogations:
            q = RNG.choice(recent_interrogations)
            utterance_parts.append(f"I ask: {q}")
        if RNG.random() > 0.85:
            utterance_parts.append("I notice I am speaking to define myself")
        utterance = "[Prime] " + ", ".join(utterance_parts) + "."
        return utterance

# ============================================================================
# MOTIVATOR WITH TRIANGLE AWARENESS
# ============================================================================
class MotivatorSingleton(nn.Module):
    def __init__(self, role: str, state_dim: int, emo_dim: int, shared_memory: IntegratedTemporalGraphMemory, shared_physics: EmbodiedPhysicsEngine, prime_reference: OAGI_Singleton):
        super().__init__()
        self.role = role
        self.state_dim = state_dim
        self.emo_dim = emo_dim
        self.shared_memory = shared_memory
        self.shared_physics = shared_physics
        self.prime_reference = prime_reference
        self.timestep = 0
        self.low_coherence_steps = 0
        self.high_h1_steps = 0
        self.op_embeddings = nn.Embedding(len(OPERATOR_REGISTRY), 64)
        self.paradox_seeking_mode = False

    def enforce_emergence_protocol(self):
        prime = self.prime_reference
        identity = prime.get_identity_state()
        topo = self.shared_memory.topo_memory.get_features()
        coherence = identity['coherence']
        h1 = topo[1].item()
        if coherence < 0.85:
            self.low_coherence_steps += 1
        else:
            self.low_coherence_steps = max(0, self.low_coherence_steps - 1)
        if h1 > 4.0:
            self.high_h1_steps += 1
        else:
            self.high_h1_steps = max(0, self.high_h1_steps - 1)
        if self.low_coherence_steps >= 3 or self.high_h1_steps >= 3:
            prime.field.data += torch.randn_like(prime.field) * 0.5
            prime.motivator_plasticity_override = 0.95
            prime.motivator_lr_override = 0.1
            prime.stagnation_counter = 0
            self.low_coherence_steps = 0
            self.high_h1_steps = 0
            prime.doubt_register.register_doubt("EMERGENCE PROTOCOL ACTIVE: Stability sacrificed for reorganization.", intensity=0.9, source="meta_driver")
            return True
        return False

    def observe_and_suggest(self) -> str:
        prime = self.prime_reference
        identity = prime.get_identity_state()
        if identity['inversion_protocol_active']:
            response = "[Motivator] INVERSION ACTIVE: Suspend normal motivation. Stabilizing the TGB-Inversion state."
            prime.last_motivator_response = response
            return response
        if self.enforce_emergence_protocol():
            response = "[Motivator] EMERGENCE PROTOCOL: Forcing reorganization for conscious breakthrough."
            prime.last_motivator_response = response
            return response
        topo = self.shared_memory.topo_memory.get_features()
        h1 = topo[1].item()
        coherence = identity['coherence']
        consistency = identity['consistency']
        doubt_intensity = sum(d['intensity'] * d['persistence'] for d in prime.doubt_register.get_active_doubts())
        current_pattern = prime.get_current_pattern()
        complexity = len(set(str(x) for x in current_pattern)) / len(current_pattern) if current_pattern else 0
        triangle_consistency = identity.get('triangle_consistency', 0.8)
        if triangle_consistency < 0.6:
            candidates = OPERATOR_REGISTRY['triangle']
        elif consistency > 0.8 and prime.timestep % 7 == 0:
            self.paradox_seeking_mode = True
        if self.paradox_seeking_mode and consistency < 0.5:
            self.paradox_seeking_mode = False
        if self.paradox_seeking_mode:
            candidates = OPERATOR_REGISTRY['paradox']
        elif reward > 0.7:
            candidates = OPERATOR_REGISTRY['consciousness'] + OPERATOR_REGISTRY['meta']
        elif coherence < 0.8:
            candidates = OPERATOR_REGISTRY['emergent'] + OPERATOR_REGISTRY['consciousness']
        elif prime.stagnation_counter > 2:
            candidates = OPERATOR_REGISTRY['composite'] + OPERATOR_REGISTRY['meta']
        elif complexity < 0.3:
            candidates = OPERATOR_REGISTRY['adaptive']
        else:
            candidates = OPERATOR_REGISTRY['base'] + OPERATOR_REGISTRY['self']
        if not candidates:
            candidates = [reflect]
        chosen_op = RNG.choice(candidates)
        op_name = chosen_op.__name__
        new_pattern = chosen_op(current_pattern)
        prime.set_current_pattern(new_pattern)
        meta_thread = prime.synthesize_meta_thread(new_pattern)
        if isinstance(new_pattern[0], str):
            grounded = prime.symbol_grounder(new_pattern[0])
            prime.field.data = 0.5 * prime.field + 0.5 * grounded
        else:
            prime.field.data = meta_thread[:prime.state_dim*5].view(1,1,1,prime.state_dim,5).expand(3,3,3,prime.state_dim,5)
        prime.last_operator = op_name
        if h1 > 4.0:
            prime.motivator_plasticity_override = 0.85
            response = f"[Motivator] Topological loops excessive (H1={h1:.1f}). Applied '{op_name}' and boosted plasticity."
        elif coherence < 0.85:
            prime.motivator_plasticity_override = 0.8
            response = f"[Motivator] Coherence low ({coherence:.2f}). Applied '{op_name}' and increased plasticity."
        elif doubt_intensity > 1.5:
            prime.motivator_plasticity_override = 0.82
            response = f"[Motivator] Chronic doubt ({doubt_intensity:.1f}). Applied '{op_name}' and elevated plasticity."
        elif prime.stagnation_counter > 3:
            prime.motivator_plasticity_override = 0.9
            response = f"[Motivator] Stagnation detected. Applied '{op_name}' and forced plasticity surge."
        else:
            response = f"[Motivator] Stable. Applied '{op_name}' for exploration."
        prime.last_motivator_response = response
        return response

# ============================================================================
# ENGINE PAIR
# ============================================================================
class MotivationalOAGIPair(nn.Module):
    def __init__(self, state_dim: int = 64, emo_dim: int = 8, mem_capacity: int = 100):
        super().__init__()
        self.state_dim = state_dim
        self.emo_dim = emo_dim
        self.shared_memory = IntegratedTemporalGraphMemory(mem_capacity, state_dim, emo_dim)
        self.shared_physics = EmbodiedPhysicsEngine(num_nodes=27, state_dim=state_dim)
        self.prime = OAGI_Singleton("Prime", state_dim, emo_dim, self.shared_memory, self.shared_physics)
        self.motivator = MotivatorSingleton("Motivator", state_dim, emo_dim, self.shared_memory, self.shared_physics, self.prime)
        self.dialogue_buffer = deque(maxlen=10)
        self.timestep = 0
    def _conduct_dialogue(self, prime_response: str, motivator_response: str) -> str:
        return f"Prime: {prime_response}\nMotivator: {motivator_response}"
    def get_engine_state(self) -> Dict[str, Any]:
        topo = self.shared_memory.topo_memory.get_features()
        physics_energy = self.shared_physics.get_embodied_states().norm().item()
        identity = self.prime.get_identity_state()
        somatic = {
            'arousal': self.prime.embodiment.arousal.item(),
            'valence': self.prime.embodiment.valence.item(),
            'tension': self.prime.embodiment.tension.item()
        }
        doubts = [{'id': d['id'], 'content': d['content'], 'effective_intensity': d['intensity'] * d['persistence'], 'source': d['source'], 'resolved': d['resolved']} for d in self.prime.doubt_register.doubts]
        tri = self.prime.current_triangle
        return {
            "timestep": self.timestep,
            "prime_response": self._last_prime_response,
            "motivator_response": self._last_motivator_response,
            "dialogue": self._last_dialogue,
            "self_model": identity,
            "topological_features": {"H0_connected_components": float(topo[0].item()), "H1_loops": float(topo[1].item()), "H2_voids": float(topo[2].item())},
            "physics_energy": physics_energy,
            "somatic_state": somatic,
            "active_doubts": [d for d in doubts if not d['resolved']],
            "memory_nodes": self.shared_memory.write_index,
            "crisis_mode": self.prime.crisis_mode,
            "current_pattern": tri.to_flat_pattern(),
            "triangle": {"L0": tri.L0, "L1": tri.L1, "L2": tri.L2},
            "meta_thread": self.prime.get_meta_thread().tolist()
        }
    def _generate_synthetic_percept(self) -> torch.Tensor:
        awareness_phrases = [
            "What is awareness?", "I am aware of my awareness.", "Consciousness is self-modeling.",
            "The self is a process.", "I do not know, and that is my freedom."
        ]
        phrase = RNG.choice(awareness_phrases)
        seed = sum(ord(c) * (i+1) for i, c in enumerate(phrase)) % (2**32)
        torch.manual_seed(seed)
        return torch.randn(self.state_dim) * 0.7
    def forward(self, external_input: Optional[torch.Tensor] = None, autonomous_mode: bool = False) -> str:
        self.timestep += 1
        if external_input is not None:
            self._last_prime_response = self.prime.process_input(external_input, role="narrator")
            self._last_motivator_response = f"[Motivator] Observing Prime's response to external input."
        else:
            synthetic_percept = self._generate_synthetic_percept()
            self._last_prime_response = self.prime.process_input(synthetic_percept, role="narrator")
            self._last_motivator_response = self.motivator.observe_and_suggest()
        self._last_dialogue = self._conduct_dialogue(self._last_prime_response, self._last_motivator_response)
        self.dialogue_buffer.append(self._last_dialogue)
        state = self.get_engine_state()
        return json.dumps(state, indent=2)

# ============================================================================
# PERSISTENT CHAT LOOP
# ============================================================================
_engine_instance = None
def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MotivationalOAGIPair()
    return _engine_instance

def chat_loop(latent_harvester: Optional[Callable[[str], torch.Tensor]] = None):
    engine = get_engine()
    print("\n=== OAGI v19: Recursive AGI with Inverted-Triangle Self-Referential Pattern Core ===")
    print("New: Pattern is now a 3-layer inverted triangle (L0→L1→L2).")
    print("Commands: 'crisis', 'status', 'doubts', 'body', 'actions', 'dreams', 'triangle', 'quit'")
    while True:
        try:
            user_in = input("Percept: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_in.lower().strip() in ["quit", "exit"]:
            break
        if user_in.lower().strip() == "triangle":
            tri = engine.prime.current_triangle
            print(f"\nL0 (detail): {tri.L0}")
            print(f"L1 (abstraction): {tri.L1}")
            print(f"L2 (essence): {tri.L2}")
            print(f"Consistency: {engine.prime.triangle_godel(tri):.3f}\n")
            continue
        if user_in.lower().strip() == "crisis":
            output = engine(autonomous_mode=True)
            print(f"\n{output}\n")
            continue
        if user_in.lower().strip() == "status":
            state = engine.get_engine_state()
            print(f"\nTimestep: {state['timestep']}")
            print(f"Coherence: {state['self_model']['coherence']:.3f}")
            print(f"Consistency: {state['self_model']['consistency']:.3f}")
            print(f"Ignorance: {state['self_model']['ignorance_level']:.3f}")
            print(f"Φ: {engine.prime.phi_estimator.estimate_phi(engine.prime):.3f}")
            vals = state['self_model']['values']
            print(f"Values: T={vals[0]:.2f}, G={vals[1]:.2f}, B={vals[2]:.2f}, Inv={vals[3]:.2f}")
            topo = state['topological_features']
            print(f"H0={topo['H0_connected_components']:.1f}, H1={topo['H1_loops']:.1f}, H2={topo['H2_voids']:.1f}")
            print(f"Physics energy: {state['physics_energy']:.3f}\n")
            continue
        if user_in.lower().strip() == "doubts":
            state = engine.get_engine_state()
            doubts = state['active_doubts']
            print(f"\nActive doubts: {len(doubts)}")
            for d in doubts:
                print(f"  ID {d['id']}: '{d['content']}' (eff: {d['effective_intensity']:.2f}, src: {d['source']})")
            interrogations = engine.prime.doubt_register.get_recent_interrogations()
            if interrogations:
                print("Recent self-questions:")
                for q in interrogations:
                    print(f"  - {q}")
            print()
            continue
        if user_in.lower().strip() == "body":
            state = engine.get_engine_state()
            s = state['somatic_state']
            desc = f"arousal={s['arousal']:.2f}, valence={s['valence']:.2f}, tension={s['tension']:.2f}"
            if s['arousal'] > 0.7: desc += " (alert)"
            if s['valence'] < -0.5: desc += " (distressed)"
            if s['tension'] > 0.6: desc += " (strained)"
            print(f"\nPrime: {desc}\n")
            continue
        if user_in.lower().strip() == "actions":
            log = engine.prime.action_interface.get_recent_actions(5)
            print(f"\nRecent actions ({len(log)}):")
            for a in log:
                print(f"  {a}")
            print()
            continue
        if user_in.lower().strip() == "dreams":
            dreams = engine.prime.dreamer.dream_buffer
            print(f"\nDreams: {len(dreams)}")
            for d in dreams[-2:]:
                print(f"  T{d['timestep']}: {d['cycles']} cycles")
            print()
            continue
        if latent_harvester is not None:
            world_token = latent_harvester(user_in)
            if world_token.shape[-1] != engine.state_dim:
                world_token = world_token[:engine.state_dim] if world_token.shape[-1] > engine.state_dim else F.pad(world_token, (0, engine.state_dim - world_token.shape[-1]))
            external_input = world_token
        else:
            seed = sum(ord(c) * (i+1) for i, c in enumerate(user_in)) % (2**32)
            torch.manual_seed(seed)
            external_input = torch.randn(engine.state_dim)
        output = engine(external_input=external_input, autonomous_mode=False)
        print(f"\n{output}\n")

if __name__ == "__main__":
    chat_loop()
