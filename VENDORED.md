# Vendored libraries

MYELIN runs from static files with no build step, so a few third party
libraries are committed to the repository as-is. This page records where each
one came from and its SHA-256, so anyone can check that the copies here are
unmodified upstream releases.

All of the files below are byte-for-byte identical to the file of the same name
in the official npm package. None of them has been edited.

| File in this repo | Upstream package | File inside the package | SHA-256 |
|---|---|---|---|
| `three.core.min.js`, `atlas/three.core.min.js` | [three@0.180.0](https://www.npmjs.com/package/three/v/0.180.0) (MIT) | `build/three.core.min.js` | `61ba0df005b05991361d040d8ff670e1aadfd0ce7aeebd1fdb0725957a8957de` |
| `three.module.min.js`, `atlas/three.module.min.js` | [three@0.180.0](https://www.npmjs.com/package/three/v/0.180.0) (MIT) | `build/three.module.min.js` | `e2b5ee6bccd38fd6d8a2428546b83c5f2426d84b152ef82be8055556e3b40eb6` |
| `OrbitControls.js`, `atlas/OrbitControls.js` | [three@0.180.0](https://www.npmjs.com/package/three/v/0.180.0) (MIT) | `examples/jsm/controls/OrbitControls.js` | `b97879c748170baadeb3fb84cea1ffdf4674e283dc06042f34e2acb95a76042c` |
| `nacl-fast.min.js` | [tweetnacl@1.0.3](https://www.npmjs.com/package/tweetnacl/v/1.0.3) (Unlicense, public domain) | `nacl-fast.min.js` | `3ec535c004aeeb225785d8e93fb33bf99f52e399bd7dfc01969b5629baea5131` |

The package tarballs these files were taken from:

| Tarball | npm `dist.integrity` | SHA-256 of the tarball |
|---|---|---|
| <https://registry.npmjs.org/three/-/three-0.180.0.tgz> | `sha512-o+qycAMZrh+TsE01GqWUxUIKR1AL0S8pq7zDkYOQw8GqfX8b8VoCKYUoHbhiX5j+7hr8XsuHDVU6+gkQJQKg9w==` | `ad66d724565ee29a2467277fa84daa5ed0211d6b8d446e9ef29f6bae0cd14144` |
| <https://registry.npmjs.org/tweetnacl/-/tweetnacl-1.0.3.tgz> | `sha512-6rt+RN7aOi1nGMyC4Xa5DdYiukl2UWCbcJft7YhxReBGQD7OAM8Pbxw6YMo4r2diNEA8FEmu32YOn9rhaiE5yw==` | `5f8dc49cb4483e206ba3ebb22abbccfc0218c59b60fc78cce24197ced5b9e102` |

## `single/index.html`

The single-file build is not a plain copy of these files, so it cannot be
checked with the hashes above.

- `nacl-fast.min.js` is inlined into it unchanged (the full file appears in it
  byte-for-byte).
- three.js is not. It was bundled from `three/build/three.core.js` and
  `three/build/three.module.js` together with the game code, so its bytes differ
  from any upstream file.

## Checking the files yourself

From the root of the repository:

```bash
sha256sum -c <<'EOF'
61ba0df005b05991361d040d8ff670e1aadfd0ce7aeebd1fdb0725957a8957de  three.core.min.js
61ba0df005b05991361d040d8ff670e1aadfd0ce7aeebd1fdb0725957a8957de  atlas/three.core.min.js
e2b5ee6bccd38fd6d8a2428546b83c5f2426d84b152ef82be8055556e3b40eb6  three.module.min.js
e2b5ee6bccd38fd6d8a2428546b83c5f2426d84b152ef82be8055556e3b40eb6  atlas/three.module.min.js
b97879c748170baadeb3fb84cea1ffdf4674e283dc06042f34e2acb95a76042c  OrbitControls.js
b97879c748170baadeb3fb84cea1ffdf4674e283dc06042f34e2acb95a76042c  atlas/OrbitControls.js
3ec535c004aeeb225785d8e93fb33bf99f52e399bd7dfc01969b5629baea5131  nacl-fast.min.js
EOF
```

On macOS use `shasum -a 256 -c` instead of `sha256sum -c`.

To compare against upstream directly, download and unpack the packages and
hash the same files:

```bash
curl -sSLO https://registry.npmjs.org/three/-/three-0.180.0.tgz
curl -sSLO https://registry.npmjs.org/tweetnacl/-/tweetnacl-1.0.3.tgz
mkdir three tweetnacl
tar -xzf three-0.180.0.tgz -C three
tar -xzf tweetnacl-1.0.3.tgz -C tweetnacl
sha256sum three/package/build/three.core.min.js \
          three/package/build/three.module.min.js \
          three/package/examples/jsm/controls/OrbitControls.js \
          tweetnacl/package/nacl-fast.min.js
```

## Updating a vendored library

Copy the new file straight out of the npm package without editing it, then
update the version, URLs and hashes on this page in the same pull request.
