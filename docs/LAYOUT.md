# Isolated layout

```
lostcoast-gc/
  engine/            → symlink to xash3d-gc
  content/
    lostcoast/       → symlink to Steam Half-Life 2/lostcoast
    hl2/             → symlink to Steam Half-Life 2/hl2
  cache/
    extracted/       → lean VPK extract
    bake/            → GCWT / GCcf / stub BSP
    stage/           → ISO staging tree
  tools/             → Source→GX converters
  scripts/           → link / bake / build / probe
  out/
    lostcoastgc.dol
    lostcoastgc.iso
  playtests/
    screenshots/
    videos/
    notes/
```

## Isolation rules

| Artifact | HL1 | dod-gc | lostcoast-gc |
|----------|-----|--------|--------------|
| DOL | `halflife1.dol` / `boot.dol` | `out/dodgc.dol` | `out/lostcoastgc.dol` |
| ISO | `OUT/xash3d-gc.iso` | `out/dodgc.iso` | `out/lostcoastgc.iso` |
| Dolphin user | `~/.local/share/xash3d-gc/dolphin` | `~/.local/share/dod-gc/dolphin` | `~/.local/share/lostcoast-gc/dolphin` |
| Playtests | engine `.ai/playtests/` | `playtests/` | `playtests/` |

Never mutate Steam trees. Never publish retail VPKs / BSPs into git.
