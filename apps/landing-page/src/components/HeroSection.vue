<script setup>
import { ref, onMounted } from 'vue'

const isVisible = ref(false)

onMounted(() => {
  setTimeout(() => { isVisible.value = true }, 100)
})
</script>

<template>
  <section id="hero" class="hero">
    <div class="hero__bg">
      <div class="hero__geo geo-1" />
      <div class="hero__geo geo-2" />
      <div class="hero__geo geo-3" />
      <div class="hero__geo geo-4" />
    </div>
    <div class="hero__gradient" />
    <div class="container hero__inner">
      <div :class="['hero__content', { visible: isVisible }]">
        <span class="hero__badge">Introducing Nexora AI v0.2</span>
        <h1 class="hero__title">Build the Future<br/>with Nexora AI</h1>
        <p class="hero__subtitle">
          A next-generation AI ecosystem designed for intelligence, speed, and scalability.
        </p>
        <div class="hero__actions">
          <a href="https://github.com/nexora-ai/nexora" target="_blank" rel="noopener" class="btn btn-primary hero__btn-primary">
            Get Started
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
            </svg>
          </a>
          <a href="https://github.com/nexora-ai/nexora?tab=readme-ov-file#nexora-ai" target="_blank" rel="noopener" class="btn btn-secondary hero__btn-secondary">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
            </svg>
            View Documentation
          </a>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.hero {
  position: relative;
  min-height: 100vh;
  display: flex;
  align-items: center;
  overflow: hidden;
  background: var(--white);
}

.hero__bg {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.hero__geo {
  position: absolute;
  border: 1.5px solid var(--gray-200);
  border-radius: 50%;
}

.geo-1 {
  width: 600px; height: 600px;
  top: -200px; right: -200px;
  animation: float 20s ease-in-out infinite;
}

.geo-2 {
  width: 400px; height: 400px;
  bottom: -100px; left: -100px;
  animation: float 25s ease-in-out infinite reverse;
}

.geo-3 {
  width: 250px; height: 250px;
  top: 20%; right: 10%;
  border: 1px solid var(--gray-100);
  animation: float 18s ease-in-out infinite;
}

.geo-4 {
  width: 180px; height: 180px;
  bottom: 30%; left: 5%;
  border: 1px solid var(--gray-100);
  animation: float 22s ease-in-out infinite reverse;
}

.hero__gradient {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 800px 400px at 80% 30%, rgba(0,0,0,0.02) 0%, transparent 70%),
    radial-gradient(ellipse 600px 300px at 20% 70%, rgba(0,0,0,0.015) 0%, transparent 70%);
  pointer-events: none;
}

@keyframes float {
  0%, 100% { transform: translate(0, 0) rotate(0deg); }
  33% { transform: translate(30px, -30px) rotate(2deg); }
  66% { transform: translate(-20px, 20px) rotate(-1deg); }
}

.hero__inner {
  position: relative;
  z-index: 2;
  padding-top: var(--nav-height);
  display: flex;
  align-items: center;
  min-height: 100vh;
}

.hero__content {
  max-width: 780px;
  opacity: 0;
  transform: translateY(40px);
  transition: all 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

.hero__content.visible {
  opacity: 1;
  transform: translateY(0);
}

.hero__badge {
  display: inline-block;
  padding: 6px 16px;
  border-radius: 50px;
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--gray-600);
  border: 1px solid var(--gray-200);
  margin-bottom: 24px;
  background: var(--gray-50);
}

.hero__title {
  font-size: clamp(2.8rem, 8vw, 5rem);
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 1.05;
  margin-bottom: 24px;
}

.hero__subtitle {
  font-size: clamp(1.1rem, 2vw, 1.375rem);
  color: var(--gray-600);
  line-height: 1.7;
  max-width: 560px;
  margin-bottom: 40px;
}

.hero__actions {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.hero__btn-primary {
  position: relative;
  overflow: hidden;
}

.hero__btn-primary::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 50%);
  transition: opacity 0.3s ease;
}

.hero__btn-primary:hover::before {
  opacity: 0;
}

.hero__btn-primary svg {
  transition: transform 0.3s ease;
}

.hero__btn-primary:hover svg {
  transform: translateX(4px);
}

.hero__btn-secondary svg {
  transition: transform 0.3s ease;
}

.hero__btn-secondary:hover svg {
  transform: translateX(-2px);
}

@media (max-width: 768px) {
  .hero__content {
    text-align: center;
    max-width: 100%;
  }
  .hero__subtitle {
    max-width: 100%;
  }
  .hero__actions {
    justify-content: center;
  }
  .geo-1 { width: 350px; height: 350px; top: -100px; right: -100px; }
  .geo-2 { width: 250px; height: 250px; }
}
</style>
