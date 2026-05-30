<script setup>
import { ref, onMounted } from 'vue'

const observer = ref(null)
const el = ref(null)

onMounted(() => {
  observer.value = new IntersectionObserver(
    (entries) => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible') }),
    { threshold: 0.2 }
  )
  if (el.value) observer.value.observe(el.value)
})
</script>

<template>
  <section id="cta" class="section cta">
    <div class="container">
      <div ref="el" class="cta__inner fade-in-scale">
        <h2 class="cta__title">Start Building with<br/>Nexora AI Today</h2>
        <p class="cta__subtitle">
          Join thousands of developers building the next generation of intelligent applications.
        </p>
        <div class="cta__actions">
          <button class="btn btn-white">
            Launch Now
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
            </svg>
          </button>
          <button class="btn btn-secondary" style="border-color: rgba(255,255,255,0.3); color: white;">
            View on GitHub
          </button>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.cta {
  background: var(--black);
  color: var(--white);
}

.cta__inner {
  text-align: center;
  padding: 80px 0;
}

.cta__title {
  font-size: clamp(2.2rem, 5vw, 3.5rem);
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 1.1;
  margin-bottom: 20px;
}

.cta__subtitle {
  font-size: 1.125rem;
  color: var(--gray-400);
  max-width: 500px;
  margin: 0 auto 40px;
  line-height: 1.7;
}

.cta__actions {
  display: flex;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
}

@media (max-width: 768px) {
  .cta__inner {
    text-align: center;
  }
}
</style>
