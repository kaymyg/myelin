# MYELIN

Collect the 165. A signature-only nerve game on a 3D human body.

You fire a nerve by signing a short plain text message with a Solana wallet.
No transaction, no fee, no server. The page calls `connect()` and
`signMessage()` and nothing else.

## What is here

| Path | What it is |
|---|---|
| `index.html` and the files beside it | The game, as a static site. Works on any static host, including GitHub Pages. |
| `single/index.html` | The same game packed into one 8.9 MB file, for hosts that take a single file (for example an IPFS website upload). |
| `atlas/` | The Nerve Signal Atlas, the anatomy explorer the game is built on, with ten animated signal routes. |
| `tools/` | The Python scripts that generated the nerve routes and packed the meshes. They need the Human Atlas source data to run. |

## How it works

1. Connect Phantom, Solflare, MetaMask (Solana account) or any Wallet Standard
   wallet. A practice key is available for trying it without a wallet.
2. Click a nerve and fire it. The wallet shows this text to sign:

   ```
   MYELIN v1
   site: <host>
   wallet: <address>
   nerve: right-median-nerve
   nonce: 7f3a91c2d4e5f601
   ts: 2026-11-12T14:02:11Z
   ```

3. The signature is checked in the browser and stored in `localStorage`.
   The nerve lights up in the colour of its speed tier.
4. Receipts can be exported as JSON and verified by anyone, offline.

## What a receipt proves

That this wallet signed this message for this nerve on this site. It does not
prove when, the time is stated by the signer's own device. Receipts cannot be
transferred to another wallet.

## Status

Live at `kaymyg.brave` (opens in the Brave browser). Connecting and firing
nerves has been tested there with a real MetaMask wallet. Phantom and Solflare
have been tested with simulated wallets only. There is no payment or support
page yet.

## Limits

- Bones and brain parts are real 3D data. Nerve routes are schematic curves
  drawn along the bones from standard anatomy, accurate to roughly a
  centimetre or two. Educational, not clinical.
- 197 of 206 bones are in the source scan. The nine missing ones (six ear
  ossicles, two inferior nasal conchae, the coccyx) unlock at 165 of 165 as
  simple hand made stand ins, labelled "reconstructed, not scanned".
- Speed tiers use one typical figure per nerve for its fastest fibres.

## Credits

- Anatomy: BodyParts3D, © The Database Center for Life Science, licensed
  under CC Attribution 4.0 International.
- Geometry simplified by the open source Human Atlas project
  (github.com/ashemag/human-atlas).
- three.js (MIT), TweetNaCl (public domain). Versions and SHA-256 hashes of
  the vendored copies are in [VENDORED.md](VENDORED.md).

The game code is released under the [MIT License](LICENSE). The anatomy data above keeps its own CC BY 4.0 license.

## Contributing

Looking for something to work on? See [HELP_WANTED.md](HELP_WANTED.md).

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get set up and run the checks,
and [SECURITY.md](SECURITY.md) for reporting vulnerabilities privately.
