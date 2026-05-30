<script setup>
import { ref, onMounted } from 'vue'

const observer = ref(null)
const el = ref(null)
const openIndex = ref(null)

const faqs = [
  {
    q: 'What is Nexora AI?',
    a: 'Nexora AI is a comprehensive, modular AI ecosystem built entirely in Rust. It provides ten specialized transformer models (NXR series), a complete training pipeline, production-grade inference engine with distributed support, and an agent framework — all designed for performance and scalability.',
  },
  {
    q: 'What hardware do I need to run Nexora?',
    a: 'Nexora supports CPU-only mode for development and small deployments. For production, GPU acceleration is available via CUDA (NVIDIA) or wgpu (Vulkan/GLES). The distributed scheduler can pool resources across multiple machines.',
  },
  {
    q: 'Can I use my own models with Nexora?',
    a: 'Yes. The inference engine supports custom model loading through the model registry. You can train models from scratch using the training pipeline, fine-tune existing NXR models, or integrate external models via the BLAA (Black Language Model API) bridge.',
  },
  {
    q: 'How does the pricing work?',
    a: 'The Starter plan is free and includes one model deployment with 1K API calls per day. Professional is $99/month for production use with advanced features. Enterprise has custom pricing with unlimited usage, dedicated support, and on-premise deployment options.',
  },
  {
    q: 'Is Nexora open source?',
    a: 'Nexora AI is available under MIT OR Apache-2.0 license. The full source code is available on GitHub. Enterprise customers get additional private support, SLAs, and custom integration assistance.',
  },
  {
    q: 'How do I get started?',
    a: 'Clone the repository, install Rust, and run cargo run --bin nexora. Our documentation covers everything from basic setup to advanced distributed deployment. The community Discord is active for questions and support.',
  },
]

function toggle(i) {
  openIndex.value = openIndex.value === i ? null : i
}

onMounted(() => {
  observer.value = new IntersectionObserver(
    (entries) => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible') }),
    { threshold: 0.05 }
  )
  if (el.value) observer.value.observe(el.value)
  document.querySelectorAll('#faq .fade-in').forEach(c => observer.value.observe(c))
})
</script>

<template>
  <section id="faq" class="section faq">
    <div class="container">
      <div ref="el" class="faq__header fade-in">
        <span class="section-label">FAQ</span>
        <h2 class="section-title">Frequently asked questions</h2>
      </div>
      <div class="faq__list">
        <div
          v-for="(item, i) in faqs"
          :key="i"
          :class="['faq__item fade-in', { 'faq__item--open': openIndex === i }]"
          :style="{ transitionDelay: `${i * 0.05}s` }"
        >
          <button class="faq__question" @click="toggle(i)">
            <span>{{ item.q }}</span>
            <svg :class="['faq__chevron', { rotated: openIndex === i }]" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="6 9 12 15 18 9"/>
            </svg>
          </button>
          <div class="faq__answer-wrapper">
            <p class="faq__answer">{{ item.a }}</p>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.faq__header {
  max-width: 600px;
  margin-bottom: 48px;
}

.faq__list {
  max-width: 720px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 0;
}

.faq__item {
  border-bottom: 1px solid var(--gray-200);
}

.faq__item:first-child {
  border-top: 1px solid var(--gray-200);
}

.faq__question {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 24px 0;
  background: none;
  border: none;
  font-family: var(--font);
  font-size: 1rem;
  font-weight: 600;
  text-align: left;
  cursor: pointer;
  color: var(--black);
  transition: color 0.2s ease;
}

.faq__question:hover {
  color: var(--gray-600);
}

.faq__chevron {
  flex-shrink: 0;
  transition: transform 0.3s ease;
  color: var(--gray-500);
}

.faq__chevron.rotated {
  transform: rotate(180deg);
}

.faq__answer-wrapper {
  display: grid;
  grid-template-rows: 0fr;
  transition: grid-template-rows 0.35s ease;
}

.faq__item--open .faq__answer-wrapper {
  grid-template-rows: 1fr;
}

.faq__answer {
  overflow: hidden;
  font-size: 0.9375rem;
  color: var(--gray-600);
  line-height: 1.7;
  padding-bottom: 0;
}

.faq__item--open .faq__answer {
  padding-bottom: 24px;
}
</style>
