use nexora_model_core::classifier_util::GenericClassifier;
use ndarray::Array2;
use std::sync::OnceLock;

pub const THREAT_CATEGORIES: [&str; 6] = [
    "injection", "xss", "auth", "crypto", "config", "network",
];
const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<GenericClassifier<{THREAT_CATEGORIES.len()}>> = OnceLock::new();

pub fn init_classifier(embed_table: Array2<f32>) {
    CLASSIFIER.set(GenericClassifier::new(embed_table, HIDDEN)).ok();
}

const THREAT_PROMPTS: &[(&str, &str)] = &[
    ("injection", "Focus on SQL/command injection, input sanitization, and parameterized queries."),
    ("xss", "Focus on cross-site scripting, content security policy, and output encoding."),
    ("auth", "Focus on authentication bypass, session management, and access control."),
    ("crypto", "Focus on cryptographic weaknesses, key management, and secure protocols."),
    ("config", "Focus on security misconfiguration, default credentials, and exposed secrets."),
    ("network", "Focus on network security, TLS, API security, and data-in-transit."),
];

/// Keyword-based threat detection — deterministic fallback for untrained ML weights.
/// Each category has a set of high-precision keywords. Score = fraction of matches.
fn keyword_threat_score(text: &str, category: &str) -> f32 {
    let lower = text.to_lowercase();
    let hits = match category {
        "injection" => {
            let kws = [
                "sql", "injection", "insert into", "drop table", "select * from",
                "union select", "';", "1=1", "exec(", "xp_cmdshell", "command injection",
                "eval(", "system(", "subprocess", "os.system", "rm -rf",
                "parameterized query", "sanitize input", "sqlmap",
            ];
            kws.iter().filter(|kw| lower.contains(*kw)).count()
        }
        "xss" => {
            let kws = [
                "xss", "cross-site", "cross site", "script>", "onerror",
                "onload", "alert(", "document.cookie", "innerhtml", "steal cookie",
                "content security policy", "csp", "output encoding",
            ];
            kws.iter().filter(|kw| lower.contains(*kw)).count()
        }
        "auth" => {
            let kws = [
                "auth", "bypass", "privilege escalation", "session hijack",
                "jwt", "oauth", "csrf", "token", "password", "login bypass",
                "authentication bypass", "access control", "rbac", "session fixation",
                "brute force", "credential stuffing",
            ];
            kws.iter().filter(|kw| lower.contains(*kw)).count()
        }
        "crypto" => {
            let kws = [
                "crypto", "encryption", "md5", "sha1", "weak cipher", "tls",
                "ssl", "man-in-the-middle", "mitm", "key exchange", "padding oracle",
                "cbc", "ecb", "ciphertext", "decrypt",
            ];
            kws.iter().filter(|kw| lower.contains(*kw)).count()
        }
        "config" => {
            let kws = [
                "config", "misconfig", "default credential", "hardcoded",
                "environment variable", ".env", "secret key", "access key",
                "s3 bucket", "public bucket", "security group", "firewall rule",
                "open port", "exposed",
            ];
            kws.iter().filter(|kw| lower.contains(*kw)).count()
        }
        "network" => {
            let kws = [
                "network", "port scan", "ddos", "mitm", "sniff", "dns spoof",
                "arp poison", "tls", "wireguard", "vpn", "proxy bypass",
                "data exfil", "c2 server", "reverse shell",
            ];
            kws.iter().filter(|kw| lower.contains(*kw)).count()
        }
        _ => 0,
    };
    let max_hits = 4u32;
    (hits as f32 / max_hits as f32).min(1.0)
}

pub fn threat_focus(threat: &str) -> &'static str {
    THREAT_PROMPTS
        .iter()
        .find(|(t, _)| *t == threat)
        .map(|(_, p)| *p)
        .unwrap_or("Focus on all security aspects including injection, XSS, auth, crypto, config, and network.")
}

pub fn detect_threat_type(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    // Phase 1: deterministic keyword-based scoring (works without ML training)
    let mut keyword_scores: Vec<(String, f32)> = THREAT_CATEGORIES
        .iter()
        .map(|cat| (cat.to_string(), keyword_threat_score(text, cat)))
        .collect();
    keyword_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // If any keyword category scores above 0.75, it's a high-confidence keyword match
    if keyword_scores[0].1 >= 0.75 {
        // Normalize to sum = 1.0 for softmax-like output
        let total: f32 = keyword_scores.iter().map(|(_, s)| s).sum();
        let total = total.max(0.1);
        return keyword_scores.into_iter().map(|(c, s)| (c, s / total)).collect();
    }

    // Phase 2: ML classifier fallback (Xavier random until trained)
    let clf = match CLASSIFIER.get() {
        Some(c) => c,
        None => return vec![(THREAT_CATEGORIES[0].to_string(), 1.0)],
    };

    // Blend: 60% ML, 40% keyword (so keyword context always influences)
    let ml_scores = clf.predict_sorted(token_ids, &THREAT_CATEGORIES, THREAT_CATEGORIES[0]);
    let blended: Vec<(String, f32)> = ml_scores
        .into_iter()
        .enumerate()
        .map(|(i, (cat, ml_score))| {
            let kw_score = keyword_scores.get(i).map(|(_, s)| *s).unwrap_or(0.0);
            (cat, ml_score * 0.6 + kw_score * 0.4)
        })
        .collect();
    let total: f32 = blended.iter().map(|(_, s)| s).sum();
    let total = total.max(0.1);
    blended.into_iter().map(|(c, s)| (c, s / total)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
fn init_cls(hidden: usize) -> GenericClassifier<6> {
        GenericClassifier::new(Array2::zeros((10, hidden)), HIDDEN)
    }

    #[test]
    fn test_detect_default_on_uninit() {
        let r = detect_threat_type("x", &[]);
        assert_eq!(r[0].0, "injection");
    }

    #[test]
    fn test_predict_empty_ids() {
        let cls = init_cls(384);
        let r = cls.predict(&[], &THREAT_CATEGORIES, "injection");
        assert_eq!(r[0].0, "injection");
    }

    #[test]
    fn test_predict_returns_all_categories() {
        let cls = init_cls(384);
        let r = cls.predict(&[0, 1], &THREAT_CATEGORIES, "injection");
        assert_eq!(r.len(), THREAT_CATEGORIES.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
