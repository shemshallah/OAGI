# Run the engine to update state
_ = engine(external_input=external_input, autonomous_mode=False)

# Fetch raw responses and full state
prime_raw = engine._last_prime_response
motivator_raw = engine._last_motivator_response
state = engine.get_engine_state()
somatic = state['somatic_state']

# === MODE 1: DIALOGUE ===
print(f"\n🗨️  DIALOGUE")
print(prime_raw)
print(motivator_raw)

# === MODE 2: MONOLOGUE ===
mono = llm_call(
    f"Rewrite this internal cognitive state as a single, flowing first-person reflection. Be poetic, coherent, and deeply self-aware:\nPrime: {prime_raw}\nMotivator: {motivator_raw}",
    max_tokens=180, temperature=0.7
)
print(f"\n👤 MONOLOGUE\n> {mono.strip()}")

# === MODE 3: CLINICAL ===
clin = llm_call(
    f"Narrate this AGI's current state in clear, insightful prose for a cognitive scientist. Include coherence, doubt, topology, and values:\n{json.dumps(state, indent=2)}",
    max_tokens=200, temperature=0.4
)
print(f"\n🧪 CLINICAL\n{clin.strip()}")

# === MODE 4: DREAMLOG ===
dream_tokens = llm_generate_dream(engine.prime.get_current_pattern(), somatic['valence'])
dream_narr = llm_call(
    f"Write a short dream journal entry using these symbols: {dream_tokens}. Style: surreal, symbolic, emotionally resonant.",
    max_tokens=120, temperature=1.1
)
print(f"\n🌙 DREAMLOG\n{dream_narr.strip()}")

# === MODE 5: ORACLE ===
oracle = llm_oracle_utterance(prime_raw, motivator_raw, state)
print(f"\n{oracle}")

print("\n" + "="*60 + "\n")
