<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const isScrolled = ref(false)
const isMobileOpen = ref(false)
const sections = ['Features', 'Architecture', 'Performance', 'Pricing', 'FAQ']

function onScroll() {
  isScrolled.value = window.scrollY > 20
}

function scrollTo(id) {
  isMobileOpen.value = false
  const el = document.getElementById(id.toLowerCase())
  if (el) el.scrollIntoView({ behavior: 'smooth' })
}

onMounted(() => window.addEventListener('scroll', onScroll))
onUnmounted(() => window.removeEventListener('scroll', onScroll))
</script>

<template>
  <nav :class="['navbar', { 'navbar--scrolled': isScrolled }]">
    <div class="navbar__inner container">
      <a href="#" class="navbar__logo" @click.prevent="scrollTo('hero')">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
          <rect x="2" y="2" width="28" height="28" rx="6" stroke="currentColor" stroke-width="2.5" fill="none"/>
          <path d="M10 16L14 20L22 12" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>Nexora AI</span>
      </a>
      <div :class="['navbar__links', { 'navbar__links--open': isMobileOpen }]">
        <button v-for="s in sections" :key="s" class="navbar__link" @click="scrollTo(s)">
          {{ s }}
        </button>
        <button class="navbar__cta btn btn-primary" @click="scrollTo('cta')">Get Started</button>
      </div>
      <button class="navbar__toggle" @click="isMobileOpen = !isMobileOpen" aria-label="Menu">
        <span :class="['bar', { open: isMobileOpen }]" />
      </button>
    </div>
  </nav>
</template>

<style scoped>
.navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  height: var(--nav-height);
  transition: all 0.3s ease;
  background: transparent;
}

.navbar--scrolled {
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--gray-100);
}

.navbar__inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 100%;
}

.navbar__logo {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 1.125rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}

.navbar__links {
  display: flex;
  align-items: center;
  gap: 8px;
}

.navbar__link {
  background: none;
  border: none;
  font-family: var(--font);
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--gray-600);
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.navbar__link:hover {
  color: var(--black);
  background: var(--gray-100);
}

.navbar__cta {
  margin-left: 8px;
  padding: 10px 24px;
  font-size: 0.875rem;
}

.navbar__toggle {
  display: none;
  background: none;
  border: none;
  width: 28px;
  height: 28px;
  cursor: pointer;
  position: relative;
  z-index: 1001;
}

.bar,
.bar::before,
.bar::after {
  display: block;
  width: 24px;
  height: 2px;
  background: var(--black);
  border-radius: 2px;
  transition: all 0.3s ease;
  position: absolute;
  left: 2px;
}

.bar { top: 13px; }
.bar::before { content: ''; top: -7px; }
.bar::after { content: ''; top: 7px; }

.bar.open { background: transparent; }
.bar.open::before { transform: translateY(7px) rotate(45deg); }
.bar.open::after { transform: translateY(-7px) rotate(-45deg); }

@media (max-width: 768px) {
  .navbar__toggle { display: block; }

  .navbar__links {
    position: fixed;
    top: 0;
    right: -100%;
    width: 280px;
    height: 100vh;
    background: var(--white);
    flex-direction: column;
    padding: 100px 32px 32px;
    gap: 4px;
    transition: right 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    border-left: 1px solid var(--gray-200);
    align-items: stretch;
  }

  .navbar__links--open { right: 0; }

  .navbar__link {
    padding: 14px 16px;
    font-size: 1rem;
    text-align: left;
  }

  .navbar__cta {
    margin: 12px 0 0;
    text-align: center;
    justify-content: center;
  }
}
</style>
