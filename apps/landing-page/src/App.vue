<script setup>
import NavBar from './components/NavBar.vue'
import HeroSection from './components/HeroSection.vue'
import FeaturesSection from './components/FeaturesSection.vue'
import ArchitectureSection from './components/ArchitectureSection.vue'
import MetricsSection from './components/MetricsSection.vue'
import ShowcaseSection from './components/ShowcaseSection.vue'
import TestimonialsSection from './components/TestimonialsSection.vue'
import PricingSection from './components/PricingSection.vue'
import FaqSection from './components/FaqSection.vue'
import CtaSection from './components/CtaSection.vue'
import FooterSection from './components/FooterSection.vue'
</script>

<template>
  <NavBar />
  <main>
    <HeroSection />
    <FeaturesSection />
    <ArchitectureSection />
    <MetricsSection />
    <ShowcaseSection />
    <TestimonialsSection />
    <PricingSection />
    <FaqSection />
    <CtaSection />
  </main>
  <FooterSection />
</template>
