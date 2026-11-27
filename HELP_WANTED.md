# Help wanted

MYELIN is open to contributions. Below is the hardening work that would make it stronger, roughly in order of impact. Pick anything, comment on (or open) an issue so others know you are on it, and see [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

Found a vulnerability instead? Please report it privately, see [SECURITY.md](SECURITY.md).


### Content Security Policy

**Medium.** Add a CSP that permits only the bundled scripts and wallet injection, and test it with the main Solana wallets.

### Record hashes of vendored libraries

**Good first issue.** Document the upstream version and SHA-256 of `three.*.min.js` and `nacl-fast.min.js` so anyone can check they are unmodified.

### Anatomy data corrections

**Good first issue.** Check nerve names and positions in `data.json` against an anatomy reference.
