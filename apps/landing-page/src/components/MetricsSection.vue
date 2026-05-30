<script setup>
import { ref, onMounted } from 'vue'

const observer = ref(null)

const metrics = [
  { value: '1,000', suffix: '+', label: 'Tokens per Second', desc: 'Target throughput with adaptive batching' },
  { value: '200', suffix: 'ms', label: 'P95 Latency', desc: 'End-to-end inference latency target' },
  { value: '10', suffix: '', label: 'NXR Models', desc: 'Specialized from Edge to Ultra tier' },
  { value: '41', suffix: '', label: 'Modular Crates', desc: 'Pure Rust workspace, 326K+ LOC' },
]

function animateValue(el, target, suffix) {
  const isMs = suffix === 'ms'
  const hasComma = target.includes(',')
  const num = parseFloat(target.replace(/,/g, '')) || 0
  const duration = 1500
  const startTime = performance.now()

  function update(now) {
    const t = Math.min((now - startTime) / duration, 1)
    const eased = 1 - Math.pow(1 - t, 3)
    const current = Math.round(eased * num)

    if (hasComma) {
      el.textContent = current.toLocaleString() + (suffix || '')
    } else if (isMs) {
      el.textContent = current + suffix
    } else {
      el.textContent = current + (suffix || '')
    }

    if (t < 1) requestAnimationFrame(update)
    else el.textContent = target + suffix
  }
  requestAnimationFrame(update)
}

onMounted(() => {
  observer.value = new IntersectionObserver(
    (entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('visible')
          const val = e.target.querySelector('.metric-value')
          const idx = parseInt(e.target.dataset.index)
          if (val && !isNaN(idx)) {
            animateValue(val, metrics[idx].value, metrics[idx].suffix)
          }
        }
      })
    },
    { threshold: 0.3 }
  )
  document.querySelectorAll('#performance .metric-card').forEach(c => observer.value.observe(c))
})
</script>

<template>
  <section id="performance" class="section metrics">
    <div class="container">
      <div class="metrics__header fade-in">
        <span class="section-label">Performance</span>
        <h2 class="section-title">Built for scale</h2>
        <p class="section-subtitle">
          From edge devices to distributed clusters — engineered for speed and reliability.
        </p>
      </div>
      <div class="metrics__grid">
        <div
          v-for="(m, i) in metrics"
          :key="i"
          class="metric-card fade-in-scale"
          :data-index="i"
        >
          <div class="metric-value">{{ m.value }}{{ m.suffix }}</div>
          <div class="metric-label">{{ m.label }}</div>
          <div class="metric-desc">{{ m.desc }}</div>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.metrics__header {
  max-width: 600px;
  margin-bottom: 64px;
}

.metrics__grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
}

.metric-card {
  text-align: center;
  padding: 48px 24px 40px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--gray-200);
  transition: all 0.3s ease;
}

.metric-card:hover {
  border-color: var(--black);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
}

.metric-value {
  font-size: clamp(2.5rem, 4vw, 3.5rem);
  font-weight: 900;
  letter-spacing: -0.04em;
  color: var(--black);
  margin-bottom: 8px;
}

.metric-label {
  font-size: 0.9375rem;
  color: var(--black);
  font-weight: 600;
  margin-bottom: 6px;
}

.metric-desc {
  font-size: 0.8125rem;
  color: var(--gray-500);
  line-height: 1.5;
  max-width: 200px;
  margin: 0 auto;
}

@media (max-width: 900px) {
  .metrics__grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
  }
  .metric-card { padding: 36px 16px 32px; }
}

@media (max-width: 400px) {
  .metrics__grid { grid-template-columns: 1fr; }
}
</style>
