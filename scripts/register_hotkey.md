# Registering a hotkey for SN68 mining

The miner needs a hotkey **registered** on netuid 68 (mainnet) or 379 (testnet). Registration costs TAO (~0.01-0.05 τ on mainnet depending on burn) but is a one-time cost.

## 1. Create the wallet (if you don't have one)

```bash
# Coldkey (holds TAO; sign one-time-only, then keep it offline)
btcli wallet new_coldkey --wallet.name <coldkey_name>

# Hotkey (used by the miner; lives on the box)
btcli wallet new_hotkey --wallet.name <coldkey_name> --wallet.hotkey <hotkey_name>
```

## 2. Fund the coldkey

Send TAO to the coldkey's SS58 address. View with:
```bash
btcli wallet inspect --wallet.name <coldkey_name>
```

For testnet 379, get free testnet TAO from the Bittensor Discord faucet (#test-tau channel).

## 3. Register on the subnet

```bash
# Mainnet (netuid 68)
btcli subnet register \
    --wallet.name <coldkey_name> \
    --wallet.hotkey <hotkey_name> \
    --netuid 68 \
    --subtensor.network finney

# Testnet (netuid 379) — much cheaper, no real TAO at stake
btcli subnet register \
    --wallet.name <coldkey_name> \
    --wallet.hotkey <hotkey_name> \
    --netuid 379 \
    --subtensor.network test
```

`btcli` will tell you the current burn (registration cost) and ask for confirmation.

## 4. Verify

```bash
btcli wallet overview --wallet.name <coldkey_name> --netuid 68
```

You should see your hotkey with a UID assigned.

## 5. Now deploy

```bash
./scripts/deploy_miner.sh \
    --rental-id <basilica-uuid> \
    --wallet <coldkey_name> \
    --hotkey <hotkey_name> \
    --netuid 68 \
    --network finney \
    --mol-surrogate models/surrogate_Q6P6W3 \
    --nb-surrogate models/nb_surrogate_Q9NZQ7
```

## Coldkey security

`scripts/deploy_miner.sh` pushes ONLY:
- `coldkeypub.txt` (just the public address — safe)
- `hotkeys/<hotkey_name>` (signed by coldkey; can submit but not transfer TAO)

The coldkey **itself** never leaves your local machine. If the Basilica box is compromised, an attacker can spam submissions but cannot move your TAO.

## Where do payouts go?

Winning epoch payouts are dispatched off-chain to your **coldkey's coldkey address** (the address corresponding to your local coldkey, not the hotkey). You verify receipts via:
```bash
btcli wallet history --wallet.name <coldkey_name>
```
or by checking the coldkey's SS58 address on Taostats.
