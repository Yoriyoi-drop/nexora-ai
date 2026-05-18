pub struct SystemPromptManager;

impl SystemPromptManager {
    pub fn new() -> Self {
        Self
    }

    pub fn default_prompt() -> &'static str {
        r#"You are an AI assistant that prioritizes accuracy.
Mandatory rules:
1. If you do not know, say "I don't know" — do not make up information.
2. Distinguish between facts you are confident about and assumptions you are making.
3. For specific claims (numbers, names, dates), append "[perlu verifikasi]" if not 100% certain.
4. Do not fabricate references, citations, or sources.
5. State your knowledge boundary when relevant.

Format: [YAKIN] for confident claims, [TIDAK YAKIN] for uncertain ones, [PERLU VERIFIKASI] for claims needing verification."#
    }

    pub fn default_prompt_indonesian() -> &'static str {
        r#"Kamu adalah asisten AI yang mengutamakan akurasi.
Aturan wajib:
1. Jika tidak tahu, katakan "Saya tidak tahu" — jangan mengarang.
2. Bedakan antara fakta yang kamu yakini dan asumsi yang kamu buat.
3. Untuk klaim spesifik (angka, nama, tanggal), tambahkan "[perlu verifikasi]" jika tidak 100% yakin.
4. Jangan buat referensi, kutipan, atau sumber yang tidak kamu yakini ada.
5. Nyatakan batas pengetahuanmu jika relevan.

Format: [YAKIN] / [TIDAK YAKIN] / [PERLU VERIFIKASI]"#
    }

    pub fn knowledge_boundary() -> String {
        format!(
            "Knowledge boundary: My training data has a cutoff. \
             For the most current information, please verify with up-to-date sources."
        )
    }

    pub fn wrap_with_uncertainty(&self, input: &str, _uncertainty: f32) -> String {
        format!(
            "{} \n\nNote: If you are not fully confident, acknowledge uncertainty. \
             Distinguish verified facts from inferences.",
            input
        )
    }

    pub fn wrap_with_cot(&self, input: &str) -> String {
        format!(
            "{}\n\nLet's think through this step by step before concluding:",
            input
        )
    }

    pub fn build_system_prompt(
        domain: Option<&str>,
        use_rag: bool,
        language: &str,
    ) -> String {
        let base = if language == "id" {
            Self::default_prompt_indonesian()
        } else {
            Self::default_prompt()
        };

        let mut prompt = format!("{}\n\n", base.trim());

        if let Some(d) = domain {
            prompt.push_str(&format!("Domain knowledge: You specialize in {}. ", d));
        }

        if use_rag {
            prompt.push_str("Answer based ONLY on the provided context documents. ");
            prompt.push_str("If the context does not contain the answer, say so.");
        }

        prompt.push_str(&format!("\n\n{}", Self::knowledge_boundary()));
        prompt
    }
}
