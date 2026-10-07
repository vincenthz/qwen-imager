//! Versioned content encryption for remote prompts and images. HTTP metadata is separate.
use anyhow::{Context as _, Result, ensure};
use cryptoxide::{chacha20poly1305::ChaCha20Poly1305, hashing::sha2::Sha256, hkdf, x25519};
use rand::{TryRngCore, rngs::OsRng};
use std::{
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

pub const CONTENT_TYPE: &str = "application/vnd.imageforger.encrypted-v1";
pub const MULTIPART_TYPE: &str = "multipart/form-data; boundary=----image-forger-boundary";
const HEADER_LEN: usize = 76;
pub const MAX_AGE: u64 = 300;

fn random<const N: usize>() -> Result<[u8; N]> {
    let mut bytes = [0; N];
    OsRng
        .try_fill_bytes(&mut bytes)
        .context("obtaining OS randomness")?;
    Ok(bytes)
}
pub fn now() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}
pub fn public_key(text: &str) -> Result<[u8; 32]> {
    let text = text.trim();
    ensure!(
        text.len() == 64 && text.bytes().all(|b| b.is_ascii_hexdigit()),
        "Server identity must be 64 hexadecimal characters"
    );
    let mut bytes = [0; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16)?;
    }
    ensure!(bytes != [0; 32], "Invalid all-zero identity");
    Ok(bytes)
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Never Debug/Serialize a private identity. GUI identities only live in memory.
pub struct Identity([u8; 32]);
impl Identity {
    pub fn generate() -> Result<Self> {
        Ok(Self(random()?))
    }
    pub fn public(&self) -> [u8; 32] {
        x25519::base(&x25519::SecretKey::from(self.0)).into()
    }
    pub fn public_hex(&self) -> String {
        hex(&self.public())
    }
    /// Exclusively create a new owner-only file, or read the existing identity.
    /// Never replace a malformed key or silently rotate a pinned identity.
    pub fn load_or_create(path: &Path) -> Result<Self> {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(path) {
            Ok(mut file) => {
                let identity = Self::generate()?;
                file.write_all(format!("imageforger-x25519-v1:{}\n", hex(&identity.0)).as_bytes())?;
                file.sync_all()?;
                Ok(identity)
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let file = std::fs::File::open(path)?;
                let metadata = file.metadata()?;
                ensure!(
                    metadata.is_file() && metadata.len() <= 128,
                    "Invalid identity key file"
                );
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    ensure!(
                        metadata.permissions().mode() & 0o077 == 0,
                        "Identity key file must be private (chmod 600)"
                    );
                }
                let mut text = String::new();
                file.take(128).read_to_string(&mut text)?;
                let key = text
                    .trim()
                    .strip_prefix("imageforger-x25519-v1:")
                    .context("Invalid identity key file format")?;
                Ok(Self(public_key(key).context("Invalid identity key file")?))
            }
            Err(error) => Err(error).context("creating identity key file"),
        }
    }
    pub fn session(&self, server: [u8; 32]) -> Result<Session> {
        let mut header = [0; HEADER_LEN];
        header[..4].copy_from_slice(b"IFC1");
        header[4..36].copy_from_slice(&self.public());
        header[36..68].copy_from_slice(&random::<32>()?);
        header[68..].copy_from_slice(&now()?.to_be_bytes());
        Session::derive(self, server, server, header)
    }
    pub fn open_request(&self, envelope: &[u8], context: &str) -> Result<(Session, Vec<u8>)> {
        ensure!(
            envelope.len() >= HEADER_LEN + 28,
            "Truncated encrypted content"
        );
        let header: [u8; HEADER_LEN] = envelope[..HEADER_LEN].try_into()?;
        ensure!(
            &header[..4] == b"IFC1",
            "Unsupported content encryption version"
        );
        let timestamp = u64::from_be_bytes(header[68..].try_into()?);
        ensure!(
            now()?.abs_diff(timestamp) <= MAX_AGE,
            "Expired encrypted request; check client and server clocks"
        );
        let peer = header[4..36].try_into()?;
        let session = Session::derive(self, peer, self.public(), header)?;
        let plaintext = session.decrypt(&session.upload, context, &envelope[HEADER_LEN..])?;
        Ok((session, plaintext))
    }
}

/// One request context, retained with its job. Directional keys and unique counters.
pub struct Session {
    header: [u8; HEADER_LEN],
    upload: [u8; 32],
    download: [u8; 32],
    sequence: AtomicU64,
}
impl Session {
    fn derive(
        identity: &Identity,
        peer: [u8; 32],
        server: [u8; 32],
        header: [u8; HEADER_LEN],
    ) -> Result<Self> {
        let shared = x25519::dh(
            &x25519::SecretKey::from(identity.0),
            &x25519::PublicKey::from(peer),
        );
        ensure!(
            shared.as_ref().iter().any(|&b| b != 0),
            "Invalid X25519 peer identity"
        );
        let mut prk = [0; 32];
        hkdf::extract::<Sha256>(&header[36..68], shared.as_ref(), &mut prk);
        let mut info = b"ImageForger content v1 X25519 HKDF-SHA256 ChaCha20-Poly1305".to_vec();
        info.extend_from_slice(&server);
        info.extend_from_slice(&header);
        let mut keys = [0; 64];
        hkdf::expand::<Sha256>(&prk, &info, &mut keys);
        Ok(Self {
            header,
            upload: keys[..32].try_into()?,
            download: keys[32..].try_into()?,
            sequence: AtomicU64::new(0),
        })
    }
    pub fn replay_id(&self) -> Vec<u8> {
        self.header.to_vec()
    }
    fn encrypt(&self, key: &[u8; 32], context: &str, plaintext: &[u8]) -> Result<Vec<u8>> {
        let sequence = self
            .sequence
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| anyhow::anyhow!("Content nonce exhausted"))?;
        let mut nonce = [0; 12];
        nonce[4..].copy_from_slice(&sequence.to_be_bytes());
        let mut ciphertext = vec![0; plaintext.len()];
        let mut tag = [0; 16];
        ChaCha20Poly1305::new(key, &nonce, context.as_bytes()).encrypt(
            plaintext,
            &mut ciphertext,
            &mut tag,
        );
        let mut result = nonce.to_vec();
        result.extend_from_slice(&ciphertext);
        result.extend_from_slice(&tag);
        Ok(result)
    }
    fn decrypt(&self, key: &[u8; 32], context: &str, envelope: &[u8]) -> Result<Vec<u8>> {
        ensure!(envelope.len() >= 28, "Truncated encrypted content");
        let end = envelope.len() - 16;
        let mut plaintext = vec![0; end - 12];
        ensure!(
            ChaCha20Poly1305::new(key, envelope[..12].try_into()?, context.as_bytes()).decrypt(
                &envelope[12..end],
                &mut plaintext,
                &envelope[end..]
            ),
            "Content authentication failed; verify the pinned server identity"
        );
        Ok(plaintext)
    }
    pub fn seal_request(&self, context: &str, plaintext: &[u8]) -> Result<Vec<u8>> {
        let mut result = self.header.to_vec();
        result.extend_from_slice(&self.encrypt(&self.upload, context, plaintext)?);
        Ok(result)
    }
    fn response_key(&self, salt: &[u8]) -> [u8; 32] {
        let mut info = b"ImageForger response v1".to_vec();
        info.extend_from_slice(salt);
        let mut key = [0; 32];
        hkdf::expand::<Sha256>(&self.download, &info, &mut key);
        key
    }
    pub fn seal_response(&self, context: &str, plaintext: &[u8]) -> Result<Vec<u8>> {
        // Fresh message keys also prevent nonce reuse after a server restart or
        // replay of a stateless identity challenge. Counters alone cannot do that.
        let salt = random::<32>()?;
        let mut result = salt.to_vec();
        result.extend_from_slice(&self.encrypt(&self.response_key(&salt), context, plaintext)?);
        Ok(result)
    }
    pub fn open_response(&self, context: &str, envelope: &[u8]) -> Result<Vec<u8>> {
        ensure!(envelope.len() >= 60, "Truncated encrypted response");
        self.decrypt(
            &self.response_key(&envelope[..32]),
            context,
            &envelope[32..],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trip_tampering_direction_context_and_wrong_pin() {
        let server = Identity::generate().unwrap();
        let client = Identity::generate().unwrap();
        let session = client.session(server.public()).unwrap();
        let body = session
            .seal_request("POST /jobs", b"private prompt and image")
            .unwrap();
        let (remote, plaintext) = server.open_request(&body, "POST /jobs").unwrap();
        assert_eq!(plaintext, b"private prompt and image");
        assert!(server.open_request(&body, "POST /identity").is_err());
        assert!(
            Identity::generate()
                .unwrap()
                .open_request(&body, "POST /jobs")
                .is_err()
        );
        let response = remote.seal_response("GET /jobs/1/image", b"png").unwrap();
        assert_eq!(
            session
                .open_response("GET /jobs/1/image", &response)
                .unwrap(),
            b"png"
        );
        assert!(
            session
                .open_response("GET /jobs/2/image", &response)
                .is_err()
        );
        assert!(
            session
                .decrypt(&session.download, "POST /jobs", &body[HEADER_LEN..])
                .is_err()
        );
        assert_ne!(
            response,
            remote.seal_response("GET /jobs/1/image", b"png").unwrap()
        );
        for i in 0..body.len() {
            let mut changed = body.clone();
            changed[i] ^= 1;
            assert!(
                server.open_request(&changed, "POST /jobs").is_err(),
                "byte {i}"
            );
        }
        for n in 0..HEADER_LEN + 28 {
            assert!(server.open_request(&body[..n], "POST /jobs").is_err());
        }
        let (restarted, _) = server.open_request(&body, "POST /jobs").unwrap();
        let after_restart = restarted
            .seal_response("GET /jobs/1/image", b"png")
            .unwrap();
        assert_ne!(response, after_restart);
        assert_eq!(
            session
                .open_response("GET /jobs/1/image", &after_restart)
                .unwrap(),
            b"png"
        );
        for i in 0..response.len() {
            let mut changed = response.clone();
            changed[i] ^= 1;
            assert!(
                session
                    .open_response("GET /jobs/1/image", &changed)
                    .is_err()
            );
        }
        let mut expired = client.session(server.public()).unwrap();
        expired.header[68..].copy_from_slice(&(now().unwrap() - MAX_AGE - 1).to_be_bytes());
        // Re-derive so this is an authentic but stale request.
        let expired =
            Session::derive(&client, server.public(), server.public(), expired.header).unwrap();
        assert!(
            server
                .open_request(
                    &expired.seal_request("POST /jobs", b"old").unwrap(),
                    "POST /jobs"
                )
                .is_err()
        );
        assert!(client.session([0; 32]).is_err());
        assert!(
            client
                .session({
                    let mut p = [0; 32];
                    p[0] = 1;
                    p
                })
                .is_err()
        );
    }
    #[test]
    fn wire_format_matches_independent_openssl_vector() {
        // Node's built-in crypto generated this vector using fixed keys 0x01/0x02,
        // request salt 0x03, response salt 0x04, timestamp 1700000000 and nonce 0.
        let vector: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/content-crypto-v1.json")).unwrap();
        let bytes = |name: &str| {
            let text = vector[name].as_str().unwrap();
            (0..text.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
                .collect::<Vec<_>>()
        };
        let server = Identity([1; 32]);
        let client = Identity([2; 32]);
        assert_eq!(server.public_hex(), vector["server_public"]);
        assert_eq!(client.public_hex(), vector["client_public"]);
        let session = Session::derive(
            &client,
            server.public(),
            server.public(),
            bytes("header").try_into().unwrap(),
        )
        .unwrap();
        assert_eq!([session.upload, session.download].concat(), bytes("keys"));
        assert_eq!(
            session
                .seal_request("POST /jobs", b"private prompt")
                .unwrap(),
            bytes("request")
        );
        assert_eq!(
            session
                .open_response("GET /jobs/1/image", &bytes("response"))
                .unwrap(),
            b"PNG test bytes"
        );
    }

    #[test]
    fn key_file_reuse_and_invalid_keys() {
        let path =
            std::env::temp_dir().join(format!("imageforger-key-{}", hex(&random::<16>().unwrap())));
        let a = Identity::load_or_create(&path).unwrap();
        let b = Identity::load_or_create(&path).unwrap();
        assert_eq!(a.public(), b.public());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
            assert!(Identity::load_or_create(&path).is_err());
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
            assert_eq!(
                Identity::load_or_create(&path).unwrap().public(),
                a.public()
            );
        }
        std::fs::write(&path, "invalid").unwrap();
        assert!(Identity::load_or_create(&path).is_err());
        std::fs::remove_file(path).unwrap();
        assert!(public_key(&"é".repeat(32)).is_err());
    }
}
