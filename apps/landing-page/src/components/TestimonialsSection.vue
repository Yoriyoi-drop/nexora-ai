<script setup>
import { ref, onMounted } from 'vue'

const observer = ref(null)
const el = ref(null)

const testimonials = [
  {
    name: 'Sarah Chen',
    role: 'CTO, TechVentures',
    text: 'Nexora AI transformed how we deploy models. The distributed architecture and prefix caching alone cut our inference costs by 60%.',
  },
  {
    name: 'Marcus Rivera',
    role: 'Lead AI Engineer, DataSync',
    text: 'The MoE gating and SACA reasoning pipeline give us production-grade accuracy without the complexity of managing multiple systems.',
  },
  {
    name: 'Elena Kowalski',
    role: 'Founder, Aether Labs',
    text: 'Building an AI startup is hard — Nexora makes it feel like cheating. The agent ecosystem alone saved us months of development.',
  },
]

onMounted(() => {
  observer.value = new IntersectionObserver(
    (entries) => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible') }),
    { threshold: 0.1 }
  )
  if (el.value) observer.value.observe(el.value)
  document.querySelectorAll('#testimonials .fade-in').forEach(c => observer.value.observe(c))
})
</script>

<template>
  <section id="testimonials" class="section testimonials">
    <div class="container">
      <div ref="el" class="testimonials__header fade-in">
        <span class="section-label">Testimonials</span>
        <h2 class="section-title">Trusted by innovators</h2>
      </div>
      <div class="testimonials__grid">
        <div
          v-for="(t, i) in testimonials"
          :key="i"
          class="testimonial-card fade-in"
          :style="{ transitionDelay: `${i * 0.15}s` }"
        >
          <div class="testimonial__avatar">
            {{ t.name.charAt(0) }}{{ t.name.split(' ')[1].charAt(0) }}
          </div>
          <p class="testimonial__text">"{{ t.text }}"</p>
          <div class="testimonial__author">
            <div class="testimonial__name">{{ t.name }}</div>
            <div class="testimonial__role">{{ t.role }}</div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.testimonials__header {
  max-width: 600px;
  margin-bottom: 64px;
}

.testimonials__grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
}

.testimonial-card {
  padding: 36px 28px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--gray-200);
  transition: all 0.3s ease;
  display: flex;
  flex-direction: column;
}

.testimonial-card:hover {
  border-color: var(--black);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
}

.testimonial__avatar {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: var(--black);
  color: var(--white);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8125rem;
  font-weight: 700;
  margin-bottom: 20px;
  letter-spacing: 0.02em;
}

.testimonial__text {
  font-size: 0.9375rem;
  color: var(--gray-700);
  line-height: 1.7;
  flex: 1;
  font-style: italic;
}

.testimonial__author {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid var(--gray-100);
}

.testimonial__name {
  font-size: 0.875rem;
  font-weight: 700;
}

.testimonial__role {
  font-size: 0.8125rem;
  color: var(--gray-500);
  margin-top: 2px;
}

@media (max-width: 768px) {
  .testimonials__grid { grid-template-columns: 1fr; }
}
</style>
