<script setup>
import { ref, onMounted } from 'vue'

const observer = ref(null)
const el = ref(null)

const plans = [
  {
    name: 'Starter',
    price: 'Free',
    desc: 'Perfect for exploring and small projects.',
    features: ['1 model deployment', '1K API calls/day', 'Community support', 'Basic analytics'],
  },
  {
    name: 'Professional',
    price: '$99',
    period: '/month',
    desc: 'For teams building production applications.',
    highlighted: true,
    features: ['10 model deployments', '100K API calls/day', 'Priority support', 'Advanced analytics', 'Custom fine-tuning', 'SSO & RBAC'],
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    desc: 'For organizations at scale.',
    features: ['Unlimited deployments', 'Unlimited API calls', 'Dedicated support', 'On-premise option', 'Custom integrations', 'SLA guarantee'],
  },
]

onMounted(() => {
  observer.value = new IntersectionObserver(
    (entries) => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible') }),
    { threshold: 0.1 }
  )
  if (el.value) observer.value.observe(el.value)
  document.querySelectorAll('#pricing .fade-in').forEach(c => observer.value.observe(c))
})
</script>

<template>
  <section id="pricing" class="section pricing">
    <div class="container">
      <div ref="el" class="pricing__header fade-in">
        <span class="section-label">Pricing</span>
        <h2 class="section-title">Simple, transparent pricing</h2>
        <p class="section-subtitle">
          Start free, scale as you grow. No hidden fees or surprise charges.
        </p>
      </div>
      <div class="pricing__grid">
        <div
          v-for="(p, i) in plans"
          :key="i"
          :class="['pricing__card fade-in', { 'pricing__card--highlighted': p.highlighted }]"
          :style="{ transitionDelay: `${i * 0.15}s` }"
        >
          <div v-if="p.highlighted" class="pricing__badge">Most Popular</div>
          <div class="pricing__name">{{ p.name }}</div>
          <div class="pricing__price">
            <span class="pricing__amount">{{ p.price }}</span>
            <span v-if="p.period" class="pricing__period">{{ p.period }}</span>
          </div>
          <p class="pricing__desc">{{ p.desc }}</p>
          <ul class="pricing__features">
            <li v-for="(f, fi) in p.features" :key="fi" class="pricing__feature">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
              {{ f }}
            </li>
          </ul>
          <button :class="['btn', p.highlighted ? 'btn-white' : 'btn-secondary', 'pricing__btn']">
            {{ p.name === 'Starter' ? 'Get Started Free' : p.name === 'Enterprise' ? 'Contact Sales' : 'Subscribe' }}
          </button>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.pricing__header {
  max-width: 600px;
  margin-bottom: 64px;
}

.pricing__grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
  align-items: start;
}

.pricing__card {
  padding: 40px 32px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--gray-200);
  position: relative;
  transition: all 0.3s ease;
}

.pricing__card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.06);
}

.pricing__card--highlighted {
  background: var(--black);
  color: var(--white);
  border-color: var(--black);
  transform: scale(1.05);
}

.pricing__card--highlighted:hover {
  transform: scale(1.05) translateY(-4px);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
}

.pricing__badge {
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%);
  background: var(--black);
  color: var(--white);
  font-size: 0.6875rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  padding: 4px 16px;
  border-radius: 50px;
  border: 1px solid var(--gray-700);
}

.pricing__card--highlighted .pricing__badge {
  background: var(--white);
  color: var(--black);
  border-color: var(--gray-300);
}

.pricing__name {
  font-size: 1rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  margin-bottom: 16px;
}

.pricing__price {
  margin-bottom: 16px;
}

.pricing__amount {
  font-size: 2.5rem;
  font-weight: 900;
  letter-spacing: -0.04em;
}

.pricing__period {
  font-size: 1rem;
  font-weight: 500;
  color: var(--gray-500);
  margin-left: 4px;
}

.pricing__card--highlighted .pricing__period {
  color: var(--gray-400);
}

.pricing__desc {
  font-size: 0.875rem;
  color: var(--gray-500);
  line-height: 1.6;
  margin-bottom: 24px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--gray-100);
}

.pricing__card--highlighted .pricing__desc {
  border-bottom-color: var(--gray-700);
  color: var(--gray-400);
}

.pricing__features {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-bottom: 32px;
  flex: 1;
}

.pricing__feature {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--gray-700);
}

.pricing__card--highlighted .pricing__feature {
  color: var(--gray-300);
}

.pricing__feature svg {
  flex-shrink: 0;
  color: var(--black);
}

.pricing__card--highlighted .pricing__feature svg {
  color: var(--white);
}

.pricing__btn {
  width: 100%;
  justify-content: center;
  padding: 14px;
}

@media (max-width: 900px) {
  .pricing__grid { grid-template-columns: 1fr; max-width: 420px; margin: 0 auto; }
  .pricing__card--highlighted { transform: none; }
  .pricing__card--highlighted:hover { transform: translateY(-4px); }
}
</style>
