<script setup>
const socials = [
  { name: 'GitHub', href: '#' },
  { name: 'Twitter', href: '#' },
  { name: 'LinkedIn', href: '#' },
  { name: 'Discord', href: '#' },
]

const links = [
  { label: 'Documentation', href: '#' },
  { label: 'API Reference', href: '#' },
  { label: 'Changelog', href: '#' },
  { label: 'Privacy', href: '#' },
  { label: 'Terms', href: '#' },
]
</script>

<template>
  <footer class="footer">
    <div class="container">
      <div class="footer__top">
        <div class="footer__brand">
          <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
            <rect x="2" y="2" width="28" height="28" rx="6" stroke="currentColor" stroke-width="2.5" fill="none"/>
            <path d="M10 16L14 20L22 12" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <span>Nexora AI</span>
        </div>
        <div class="footer__links">
          <a v-for="l in links" :key="l.label" :href="l.href" class="footer__link">{{ l.label }}</a>
        </div>
        <div class="footer__social">
          <a v-for="s in socials" :key="s.name" :href="s.href" class="footer__social-link">{{ s.name }}</a>
        </div>
      </div>
      <div class="footer__bottom">
        <p>&copy; {{ new Date().getFullYear() }} Nexora AI. All rights reserved. MIT OR Apache-2.0 License.</p>
      </div>
    </div>
  </footer>
</template>

<style scoped>
.footer {
  padding: 48px 0 32px;
  border-top: 1px solid var(--gray-200);
  background: var(--gray-50);
}

.footer__top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 24px;
  margin-bottom: 32px;
}

.footer__brand {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1rem;
  font-weight: 700;
}

.footer__links {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
}

.footer__link {
  font-size: 0.875rem;
  color: var(--gray-600);
  transition: color 0.2s ease;
}

.footer__link:hover {
  color: var(--black);
}

.footer__social {
  display: flex;
  gap: 16px;
}

.footer__social-link {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--gray-600);
  transition: color 0.2s ease;
}

.footer__social-link:hover {
  color: var(--black);
}

.footer__bottom {
  padding-top: 24px;
  border-top: 1px solid var(--gray-200);
}

.footer__bottom p {
  font-size: 0.8125rem;
  color: var(--gray-500);
}

@media (max-width: 640px) {
  .footer__top {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
