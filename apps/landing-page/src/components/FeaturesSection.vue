<script setup>
import { ref, onMounted } from 'vue'

const observer = ref(null)
const el = ref(null)

const features = [
  {
    title: 'Advanced AI Models',
    desc: 'Ten specialized NXR transformer models — from edge to ultra — each with real MLP classifiers and native subsystem integration.',
    items: [
      '10 models: Omnis, Aether, Axiom, Genesis, Nexum, Vortex, Spectra, Cipher, Kronos, Swift',
      'Hidden dims 128–768, layers 2–16, heads 4–12',
      'Real MLP classifiers (emotion, domain, code, threat, temporal, task, quality, complexity, style)',
      'MoE gating: 8 experts, top-2 routing, load balancing loss',
      'SACA 6-phase reasoning pipeline & Oracle code verifiers',
    ],
  },
  {
    title: 'High Performance Infrastructure',
    desc: 'Distributed cluster with gossip-based node discovery, continuous batching, paged KV cache with prefix sharing, and dual GPU backend.',
    items: [
      'Continuous batching engine: up to 32 sequences per batch',
      'Paged KV cache: 16-token blocks, 65K max blocks, F16 storage',
      'Prefix DAG sharing: copy-on-write, automatic block-level sharing',
      'GPU auto-detection: CUDA (NVIDIA) + wgpu (Vulkan/GLES) fallback',
      'Distributed cluster: gossip protocol, load-aware routing, 1s heartbeat',
    ],
  },
  {
    title: 'Developer-Friendly APIs',
    desc: 'Clean REST API, streaming SSE endpoints, agent ecosystem with planner-worker hierarchy, and a powerful CLI for every workflow.',
    items: [
      'REST API: /generate, /chat, /analyze, /codegen, /agents, /plans',
      'Streaming SSE: real-time token generation with [DONE] sentinel',
      'CLI: train, evaluate, infer, collect-data, health, config, load-checkpoint',
      'Agent ecosystem: planner–worker hierarchy, plan dispatch, 8 step types',
      'HuggingFace live data: train langsung dari datasets-server',
    ],
  },
]

onMounted(() => {
  observer.value = new IntersectionObserver(
    (entries) => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible') }),
    { threshold: 0.1 }
  )
  if (el.value) observer.value.observe(el.value)
  document.querySelectorAll('#features .feature-card').forEach(c => observer.value.observe(c))
})
</script>

<template>
  <section id="features" class="section features">
    <div class="container">
      <div ref="el" class="features__header fade-in">
        <span class="section-label">Features</span>
        <h2 class="section-title">Everything you need to<br/>build intelligent systems</h2>
        <p class="section-subtitle">
          From training to production, Nexora AI provides a complete toolkit for modern AI development.
        </p>
      </div>
      <div class="features__grid">
        <div
          v-for="(f, i) in features"
          :key="i"
          class="feature-card fade-in"
          :style="{ transitionDelay: `${i * 0.15}s` }"
        >
          <div class="feature-card__icon">
            <svg v-if="i === 0" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2a4 4 0 0 1 4 4c0 2-2 4-4 6-2-2-4-4-4-6a4 4 0 0 1 4-4z"/><circle cx="12" cy="16" r="4"/>
              <path d="M12 20v2"/><path d="M8 22h8"/>
            </svg>
            <svg v-if="i === 1" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/>
              <line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/>
            </svg>
            <svg v-if="i === 2" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
            </svg>
          </div>
          <h3 class="feature-card__title">{{ f.title }}</h3>
          <p class="feature-card__desc">{{ f.desc }}</p>
          <ul class="feature-card__list">
            <li v-for="(item, ii) in f.items" :key="ii">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
              {{ item }}
            </li>
          </ul>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.features__header {
  max-width: 700px;
  margin-bottom: 64px;
}

.features__grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
}

.feature-card {
  padding: 40px 32px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--gray-200);
  background: var(--white);
  transition: all 0.3s ease;
  display: flex;
  flex-direction: column;
}

.feature-card:hover {
  border-color: var(--black);
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.06);
}

.feature-card__icon {
  width: 52px;
  height: 52px;
  border-radius: 14px;
  border: 1.5px solid var(--gray-200);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
  color: var(--black);
  transition: all 0.3s ease;
}

.feature-card:hover .feature-card__icon {
  border-color: var(--black);
  background: var(--gray-50);
}

.feature-card__title {
  font-size: 1.25rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 10px;
}

.feature-card__desc {
  font-size: 0.875rem;
  color: var(--gray-600);
  line-height: 1.7;
  margin-bottom: 20px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--gray-100);
}

.feature-card__list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: auto;
}

.feature-card__list li {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  font-size: 0.8125rem;
  color: var(--gray-700);
  line-height: 1.5;
}

.feature-card__list li svg {
  flex-shrink: 0;
  margin-top: 3px;
  color: var(--black);
}

@media (max-width: 900px) {
  .features__grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 600px) {
  .features__grid { grid-template-columns: 1fr; }
  .features__header { text-align: center; }
}
</style>
