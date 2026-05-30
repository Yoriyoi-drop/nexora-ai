<script setup>
import { ref, onMounted } from 'vue'

const observer = ref(null)
const el = ref(null)

const layers = [
  {
    id: 'gateway',
    label: 'API & Agents',
    desc: 'Developer-facing interfaces',
    color: '#000',
    nodes: [
      { id: 'gateway', label: 'API Gateway', desc: 'REST, streaming, auth, metrics, rate limiting', icon: 'M16 18L22 12 16 6M8 6L2 12 6 18' },
      { id: 'agent', label: 'Agent Framework', desc: 'Planner-worker hierarchy, plan dispatch, steps', icon: 'M12 2L2 7 12 12 22 7 12 2zM2 17L12 22 22 17M2 12L12 17 22 12' },
    ],
  },
  {
    id: 'services',
    label: 'Core Services',
    desc: 'Intelligence & computation',
    color: '#222',
    nodes: [
      { id: 'inference', label: 'Inference Engine', desc: 'Continuous batching, Paged KV cache, prefix sharing', icon: 'M9.75 17L9 20 8 21M15 13L18 10 20 10 21 12M5 13L8 10 10 10 12 12' },
      { id: 'training', label: 'Training System', desc: 'AdamW, gradient accum., checkpoint, MoE gating', icon: 'M12 6V12L15 15M22 12C22 17.5228 17.5228 22 12 22 6.47715 22 2 17.5228 2 12 2 6.47715 6.47715 2 12 2 17.5228 2 22 6.47715 22 12Z' },
      { id: 'models', label: 'NXR Models', desc: '10 specialized transformers, MoE, multimodal, SACA', icon: 'M20 7L9 18 4 13' },
    ],
  },
  {
    id: 'foundation',
    label: 'Foundation Layer',
    desc: 'Core infrastructure & runtime',
    color: '#444',
    nodes: [
      { id: 'core', label: 'Core Engine', desc: 'Controller, types, execution, async runtime', icon: 'M4 8L12 3 20 8 20 16 12 21 4 16 4 8Z' },
      { id: 'runtime', label: 'Distributed Runtime', desc: 'Gossip protocol, load balancing, cluster', icon: 'M5 12H19M12 5L19 12 12 19M5 12L12 19 5 12z' },
      { id: 'storage', label: 'Memory & Storage', desc: '4-layer memory, Paged KV cache, Prefix DAG', icon: 'M22 12H2M12 2C6.477 2 2 6.477 2 12 2 17.523 6.477 22 12 22 17.523 22 22 17.523 22 12Z' },
    ],
  },
]

const flows = [
  { x1: 24, y1: 1, x2: 24, y2: 2 },
  { x1: 50, y1: 1, x2: 50, y2: 2 },
  { x1: 76, y1: 1, x2: 76, y2: 2 },
  { x1: 17, y1: 2, x2: 17, y2: 3 },
  { x1: 50, y1: 2, x2: 50, y2: 3 },
  { x1: 83, y1: 2, x2: 83, y2: 3 },
]

onMounted(() => {
  observer.value = new IntersectionObserver(
    (entries) => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible') }),
    { threshold: 0.1 }
  )
  if (el.value) observer.value.observe(el.value)
  document.querySelectorAll('#architecture .anim-layer, #architecture .anim-card').forEach(c => observer.value.observe(c))
})
</script>

<template>
  <section id="architecture" class="section arch">
    <div class="container">
      <div ref="el" class="arch__header fade-in">
        <span class="section-label">Architecture</span>
        <h2 class="section-title">Modular by design</h2>
        <p class="section-subtitle">
          A layered ecosystem built for scalability, reliability, and performance.
        </p>
      </div>

      <div class="arch__stack">
        <div
          v-for="(layer, li) in layers"
          :key="layer.id"
          class="arch__layer anim-layer"
          :style="{ transitionDelay: `${li * 0.1}s` }"
        >
          <div class="arch__layer-header">
            <div class="arch__layer-label">{{ layer.label }}</div>
            <div class="arch__layer-desc">{{ layer.desc }}</div>
          </div>
          <div class="arch__layer-nodes">
            <div
              v-for="(node, ni) in layer.nodes"
              :key="node.id"
              class="arch__card anim-card"
              :style="{ transitionDelay: `${li * 0.1 + ni * 0.08}s` }"
            >
              <div class="arch__card-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                  <path :d="node.icon" />
                </svg>
              </div>
              <div class="arch__card-label">{{ node.label }}</div>
              <div class="arch__card-desc">{{ node.desc }}</div>
            </div>
          </div>
        </div>

        <div class="arch__connectors">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none">
            <line
              v-for="(f, i) in flows"
              :key="i"
              :x1="f.x1" :y1="f.y1" :x2="f.x2" :y2="f.y2"
              stroke="var(--gray-300)" stroke-width="0.2" stroke-dasharray="1"
              class="arch__flow-line"
            />
          </svg>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.arch__header {
  max-width: 600px;
  margin-bottom: 64px;
}

.arch__stack {
  position: relative;
  max-width: 880px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 0;
}

.arch__layer {
  padding: 32px 32px 36px;
  border: 1px solid var(--gray-200);
  border-radius: var(--radius-lg);
  background: var(--white);
  position: relative;
  z-index: 1;
  transition: all 0.3s ease;
}

.arch__layer + .arch__layer {
  margin-top: -1px;
  border-top-left-radius: 0;
  border-top-right-radius: 0;
}

.arch__layer:first-child {
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
}

.arch__layer:hover {
  border-color: var(--black);
  z-index: 2;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
}

.arch__layer-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--gray-100);
}

.arch__layer-label {
  font-size: 0.8125rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--gray-500);
}

.arch__layer-desc {
  font-size: 0.8125rem;
  color: var(--gray-400);
}

.arch__layer-nodes {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
}

.arch__card {
  padding: 20px;
  border-radius: var(--radius-md);
  border: 1px solid var(--gray-100);
  transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
  cursor: default;
}

.arch__card:hover {
  border-color: var(--black);
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
  background: var(--gray-50);
}

.arch__card-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  border: 1.5px solid var(--gray-200);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 16px;
  color: var(--gray-700);
  transition: all 0.3s ease;
}

.arch__card:hover .arch__card-icon {
  border-color: var(--black);
  color: var(--black);
  background: var(--white);
}

.arch__card-label {
  font-size: 0.9375rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  margin-bottom: 6px;
}

.arch__card-desc {
  font-size: 0.78125rem;
  color: var(--gray-500);
  line-height: 1.5;
}

.arch__connectors {
  display: none;
}

.anim-layer,
.anim-card {
  opacity: 0;
  transform: translateY(20px);
  transition: opacity 0.6s ease, transform 0.6s ease;
}

.anim-layer.visible,
.anim-card.visible {
  opacity: 1;
  transform: translateY(0);
}

@media (max-width: 768px) {
  .arch__layer {
    padding: 24px 20px 28px;
  }

  .arch__layer-nodes {
    grid-template-columns: 1fr;
  }

  .arch__layer-header {
    flex-direction: column;
    gap: 4px;
  }
}
</style>
