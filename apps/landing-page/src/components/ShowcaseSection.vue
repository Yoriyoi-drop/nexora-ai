<script setup>
import { ref, onMounted } from 'vue'

const observer = ref(null)
const el = ref(null)

const cards = [
  {
    title: 'Model Training',
    desc: 'Full training pipeline with real-time monitoring, gradient accumulation, and automatic checkpointing.',
    features: ['AdamW optimizer with cosine decay', 'Gradient accumulation & clipping', 'Automatic .safetensors checkpointing', 'Mixed precision GPU training'],
    icon: 'M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z',
    chart: [65, 40, 80, 55, 90, 45, 75, 60, 85, 50, 70, 55],
  },
  {
    title: 'Inference Engine',
    desc: 'Production-grade serving with continuous batching, paged KV cache, and distributed routing.',
    features: ['Continuous batching (up to 32 seq)', 'Paged KV cache with prefix sharing', 'CUDA + wgpu GPU acceleration', 'Distributed cluster routing'],
    icon: 'M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z',
    chart: [80, 75, 90, 85, 95, 88, 92, 87, 96, 90, 93, 88],
  },
  {
    title: 'Agent Ecosystem',
    desc: 'Multi-agent orchestration with planner-worker hierarchy and inference engine integration.',
    features: ['Planner-worker agent hierarchy', 'Automatic plan dispatch & execution', 'REST API for agent management', 'Real inference engine integration'],
    icon: 'M12 2l-2 7h-7l5.5 5L7 22l5-4.5L17 22l-1.5-8L21 9h-7l-2-7z',
    chart: [85, 70, 90, 60, 95, 75, 88, 78, 92, 82, 86, 72],
  },
  {
    title: 'Model Architecture',
    desc: 'Ten specialized NXR transformer models with real MLP classifiers and MoE gating.',
    features: ['10 models from Edge (128) to Ultra (768)', 'Real MLP classifiers per model', 'MoE gating (8 experts, top-2)', 'SACA reasoning & code verifiers'],
    icon: 'M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z',
    chart: [50, 60, 45, 70, 55, 75, 60, 80, 65, 72, 58, 68],
  },
]

const activeIndex = ref(0)

function prev() {
  activeIndex.value = (activeIndex.value - 1 + cards.length) % cards.length
}

function next() {
  activeIndex.value = (activeIndex.value + 1) % cards.length
}

onMounted(() => {
  observer.value = new IntersectionObserver(
    (entries) => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible') }),
    { threshold: 0.1 }
  )
  if (el.value) observer.value.observe(el.value)
  document.querySelectorAll('#showcase .anim-card').forEach(c => observer.value.observe(c))
})
</script>

<template>
  <section id="showcase" class="section showcase">
    <div class="container">
      <div ref="el" class="showcase__header fade-in">
        <span class="section-label">Showcase</span>
        <h2 class="section-title">Experience the platform</h2>
        <p class="section-subtitle">
          Premium tools designed for modern AI workflows.
        </p>
      </div>

      <div class="showcase__carousel">
        <button class="showcase__arrow showcase__arrow--prev" @click="prev" aria-label="Previous">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="15 18 9 12 15 6"/>
          </svg>
        </button>

        <div class="showcase__viewport">
          <div
            v-for="(card, i) in cards"
            :key="i"
            :class="['showcase__card', { active: activeIndex === i, prev: i === (activeIndex - 1 + cards.length) % cards.length, next: i === (activeIndex + 1) % cards.length }]"
          >
            <div class="showcase__card-body">
              <div class="showcase__card-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                  <path :d="card.icon" />
                </svg>
              </div>
              <h3 class="showcase__card-title">{{ card.title }}</h3>
              <p class="showcase__card-desc">{{ card.desc }}</p>
              <ul class="showcase__card-features">
                <li v-for="(f, fi) in card.features" :key="fi">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  {{ f }}
                </li>
              </ul>
            </div>
            <div class="showcase__card-chart">
              <div class="showcase__chart-vis">
                <svg :viewBox="`0 0 ${card.chart.length} 100`" preserveAspectRatio="none">
                  <rect
                    v-for="(v, vi) in card.chart"
                    :key="vi"
                    :x="vi" y="0" :width="0.7"
                    :height="v"
                    :transform="`translate(0, ${100 - v})`"
                    fill="var(--black)"
                    :opacity="0.08 + (v / 100) * 0.12"
                    rx="0.3"
                  />
                </svg>
                <div class="showcase__chart-label">Performance Index</div>
              </div>
            </div>
          </div>
        </div>

        <button class="showcase__arrow showcase__arrow--next" @click="next" aria-label="Next">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="9 18 15 12 9 6"/>
          </svg>
        </button>
      </div>

      <div class="showcase__dots">
        <button
          v-for="(_, i) in cards"
          :key="i"
          :class="['showcase__dot', { active: activeIndex === i }]"
          @click="activeIndex = i"
          :aria-label="`Go to slide ${i + 1}`"
        />
      </div>
    </div>
  </section>
</template>

<style scoped>
.showcase__header {
  max-width: 600px;
  margin-bottom: 48px;
}

.showcase__carousel {
  display: flex;
  align-items: center;
  gap: 16px;
  max-width: 1000px;
  margin: 0 auto;
}

.showcase__arrow {
  flex-shrink: 0;
  width: 44px;
  height: 44px;
  border-radius: 50%;
  border: 1.5px solid var(--gray-200);
  background: var(--white);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  color: var(--gray-600);
}

.showcase__arrow:hover {
  border-color: var(--black);
  color: var(--black);
  background: var(--gray-50);
}

.showcase__viewport {
  flex: 1;
  overflow: hidden;
  position: relative;
  min-height: 420px;
}

.showcase__card {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  border-radius: var(--radius-lg);
  border: 1px solid var(--gray-200);
  background: var(--white);
  opacity: 0;
  transform: translateX(40px) scale(0.96);
  transition: all 0.45s cubic-bezier(0.16, 1, 0.3, 1);
  pointer-events: none;
  overflow: hidden;
}

.showcase__card.active {
  opacity: 1;
  transform: translateX(0) scale(1);
  pointer-events: auto;
  z-index: 2;
}

.showcase__card.prev {
  opacity: 0;
  transform: translateX(-40px) scale(0.96);
  z-index: 1;
}

.showcase__card.next {
  opacity: 0;
  transform: translateX(40px) scale(0.96);
  z-index: 1;
}

.showcase__card-body {
  padding: 36px 32px;
  display: flex;
  flex-direction: column;
}

.showcase__card-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  border: 1.5px solid var(--gray-200);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 20px;
  color: var(--black);
}

.showcase__card-title {
  font-size: 1.375rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 10px;
}

.showcase__card-desc {
  font-size: 0.875rem;
  color: var(--gray-600);
  line-height: 1.7;
  margin-bottom: 20px;
}

.showcase__card-features {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: auto;
}

.showcase__card-features li {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.8125rem;
  font-weight: 500;
  color: var(--gray-700);
}

.showcase__card-features li svg {
  flex-shrink: 0;
  color: var(--black);
}

.showcase__card-chart {
  border-left: 1px solid var(--gray-100);
  background: var(--gray-50);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 36px 28px;
}

.showcase__chart-vis {
  width: 100%;
  text-align: center;
}

.showcase__chart-vis svg {
  width: 100%;
  height: 160px;
}

.showcase__chart-label {
  font-size: 0.6875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--gray-400);
  margin-top: 12px;
}

.showcase__dots {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin-top: 24px;
}

.showcase__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  border: none;
  background: var(--gray-300);
  cursor: pointer;
  transition: all 0.3s ease;
  padding: 0;
}

.showcase__dot.active {
  background: var(--black);
  width: 24px;
  border-radius: 4px;
}

.showcase__dot:hover:not(.active) {
  background: var(--gray-500);
}

@media (max-width: 768px) {
  .showcase__card {
    grid-template-columns: 1fr;
  }

  .showcase__card-chart {
    display: none;
  }

  .showcase__card-body {
    padding: 28px 24px;
  }

  .showcase__arrow {
    width: 36px;
    height: 36px;
  }

  .showcase__arrow--prev { display: none; }
  .showcase__arrow--next { display: none; }

  .showcase__viewport {
    min-height: 340px;
  }
}
</style>
