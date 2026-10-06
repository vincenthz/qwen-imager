//! Small structured diagnostics shared by the service and desktop frontend.
use std::time::{SystemTime, UNIX_EPOCH};

pub fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// Remove known secrets before limiting output or escaping control characters.
pub fn redact(message: &str, secrets: &[&str]) -> String {
    let mut message = message.to_owned();
    for secret in secrets.iter().filter(|secret| !secret.is_empty()) {
        message = message.replace(secret, "[REDACTED]");
    }
    message = scrub_urls(&message);
    let mut clean: String = message
        .chars()
        .take(8192)
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
        .collect();
    if message.chars().count() > 8192 {
        clean.push_str("… [truncated]");
    }
    clean
}

/// Display a target without URL user-info, query credentials, or fragments.
fn raw_endpoint(url: &str) -> String {
    let url = url.split(['?', '#']).next().unwrap_or(url);
    let safe = if let Some((scheme, rest)) = url.split_once("://") {
        let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
        let authority = authority.rsplit('@').next().unwrap_or(authority);
        format!(
            "{scheme}://{authority}{}",
            if path.is_empty() {
                String::new()
            } else {
                format!("/{path}")
            }
        )
    } else {
        url.to_owned()
    };
    safe
}

pub fn endpoint(url: &str) -> String {
    redact(&raw_endpoint(url), &[])
}

fn scrub_urls(message: &str) -> String {
    let mut result = String::new();
    let mut remaining = message;
    while let Some(start) = remaining
        .find("http://")
        .into_iter()
        .chain(remaining.find("https://"))
        .min()
    {
        result.push_str(&remaining[..start]);
        let url = &remaining[start..];
        let end = url
            .char_indices()
            .find(|(_, c)| c.is_whitespace() || matches!(c, '\"' | '\'' | '<' | '>'))
            .map_or(url.len(), |(i, _)| i);
        result.push_str(&raw_endpoint(&url[..end]));
        remaining = &url[end..];
    }
    result.push_str(remaining);
    result
}

pub fn log(level: &str, event: &str, fields: serde_json::Value) {
    eprintln!(
        "{}",
        serde_json::json!({"timestamp_ms": timestamp_ms(), "level": level, "event": event, "details": fields})
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostics_strip_credentials_and_bound_untrusted_text() {
        assert_eq!(
            endpoint("https://user:secret@host:8443/api?token=secret#private"),
            "https://host:8443/api"
        );
        assert_eq!(
            redact("failed with Bearer secret\u{1b}", &["secret"]),
            "failed with Bearer [REDACTED]"
        );
        assert!(redact(&"x".repeat(9000), &[]).ends_with("[truncated]"));
        assert_eq!(
            redact(
                "fetch https://user:secret@host/file?signature=private failed",
                &[]
            ),
            "fetch https://host/file failed"
        );
    }
}
